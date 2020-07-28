#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
//#include "callback_metric.h"

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static uint64_t kernelDuration;

typedef struct MetricData_st {
    CUpti_EventGroupSet *event_groups;
    uint32_t num_events;
    CUpti_EventID *event_ids;
    uint64_t *event_values;
} MetricData_t;

typedef struct MetricData_Mul_st {
    CUpti_SubscriberHandle subscriber;
    CUdevice device;
    CUcontext context;
    MetricData_t *metric_datas;
    uint32_t metric_num;
    CUpti_MetricValue *metric_values;
    CUpti_MetricID *metric_ids;
    char **metric_names;
    int metric_passes;
    int current_pass;
} MetricData_Mul_t;

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *rawBuffer;

  *size = 16 * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  CUpti_ActivityKernel5 *kernel;

  //since we launched only 1 kernel, we should have only 1 kernel record
  CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

  kernel = (CUpti_ActivityKernel5 *)record;
  if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    fprintf(stderr, "Error: expected kernel activity record, got %d\n", (int)kernel->kind);
    exit(-1);
  }

  kernelDuration = kernel->end - kernel->start;
  free(buffer);
}


void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
              	         CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    MetricData_Mul_t *metric_data_mul = (MetricData_Mul_t*)userdata;
    unsigned int i, j, k;

    // This callback is enabled only for launch so we shouldn't see
    // anything else.
    if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
          (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
        printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
        exit(-1);
    }

    // on entry, enable all the event groups being collected this pass,
    // for metrics we collect for all instances of the event
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        cudaDeviceSynchronize();
	int current_pass = metric_data_mul->current_pass;
	MetricData_t metric_data = metric_data_mul->metric_datas[current_pass];
        if(current_pass >= metric_data_mul->metric_passes)
            return;
        CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                           CUPTI_EVENT_COLLECTION_MODE_KERNEL));
        for (i = 0; i < metric_data.event_groups->numEventGroups; i++) {
            uint32_t all = 1;
            CUPTI_CALL(cuptiEventGroupSetAttribute(metric_data.event_groups->eventGroups[i],
                                             CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                             sizeof(all), &all));
            CUPTI_CALL(cuptiEventGroupEnable(metric_data.event_groups->eventGroups[i]));
        }
    }

    // on exit, read and record event values
    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        cudaDeviceSynchronize();
        // for each group, read the event values from the group and record
        // in metricData
        int current_pass = metric_data_mul->current_pass;
        MetricData_t metric_data = metric_data_mul->metric_datas[current_pass];
	int running_sum = 0;
        for (i = 0; i < metric_data.event_groups->numEventGroups; i++) {
            CUpti_EventGroup group = metric_data.event_groups->eventGroups[i];
            CUpti_EventDomainID group_domain;
            uint32_t numEvents, numInstances, numTotalInstances;
            CUpti_EventID *eventIds;
            size_t groupDomainSize = sizeof(group_domain);
            size_t numEventsSize = sizeof(numEvents);
            size_t numInstancesSize = sizeof(numInstances);
            size_t numTotalInstancesSize = sizeof(numTotalInstances);
            uint64_t *values, normalized, sum;
            size_t valuesSize, eventIdsSize;

            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                 CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                                 &groupDomainSize, &group_domain));
            CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metric_data_mul->device, group_domain,
                                                 CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                 &numTotalInstancesSize, &numTotalInstances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                 CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                                 &numInstancesSize, &numInstances));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                 CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                                 &numEventsSize, &numEvents));
            eventIdsSize = numEvents * sizeof(CUpti_EventID);
            eventIds = (CUpti_EventID *)malloc(eventIdsSize);
            CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                                 CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                                 &eventIdsSize, eventIds));

            valuesSize = sizeof(uint64_t) * numInstances;
            values = (uint64_t *)malloc(valuesSize);

	    for(j = 0; j < numEvents; j++) {
          	CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                     eventIds[j], &valuesSize, values));
		sum = 0;
		for(k = 0; k < numInstances; k++) {
		    sum += values[k];
		}
		normalized = (sum * numTotalInstances) / numInstances;
		metric_data.event_ids[running_sum + j] = eventIds[j];
		metric_data.event_values[running_sum + j] = normalized;
	    }
	    running_sum += numEvents;
	    free(values);
	    free(eventIds);
      	}
    	for (i = 0; i < metric_data.event_groups->numEventGroups; i++)
      		CUPTI_CALL(cuptiEventGroupDisable(metric_data.event_groups->eventGroups[i]));
	metric_data_mul->current_pass++;

    }
}

#ifndef __CUPTI_PROFILER_NAME_SHORT
    #define __CUPTI_PROFILER_NAME_SHORT 128
#endif
char **availabel_metrics(CUdevice device, uint32_t *numMetric)
{
    char **metric_names;
    size_t size;
    //char metricName[__CUPTI_PROFILER_NAME_SHORT];
    CUpti_MetricValueKind metricKind;
    CUpti_MetricID *metricIdArray;

    CUPTI_CALL(cuptiDeviceGetNumMetrics(device, numMetric));
    size = sizeof(CUpti_MetricID) * (*numMetric);
    metricIdArray = (CUpti_MetricID*)malloc(size);
    metric_names = (char **)malloc(sizeof(char *) * (*numMetric));
    for(int i = 0; i < (*numMetric); i++) {
        metric_names[i] = (char *)malloc(sizeof(char) * __CUPTI_PROFILER_NAME_SHORT);
    }

    CUPTI_CALL(cuptiDeviceEnumMetrics(device, &size, metricIdArray));

    for(int i = 0; i < (*numMetric); i++) {
            size = __CUPTI_PROFILER_NAME_SHORT;
            CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i],
                    CUPTI_METRIC_ATTR_NAME, &size, (void *) metric_names[i]));
            size = sizeof(CUpti_MetricValueKind);
            CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i],
                    CUPTI_METRIC_ATTR_VALUE_KIND, &size, (void *)& metricKind));
            //if ((metricKind == CUPTI_METRIC_VALUE_KIND_THROUGHPUT)
                //|| (metricKind == CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL)) {
		
            //printf("Metric %s cannot be profiled as metric requires GPU time duration for kernel run.\n", metric_names[i]);
            //}
    }
    free(metricIdArray);
    return metric_names;
}

void print_metric(CUpti_MetricID id, CUpti_MetricValue value)
{
    CUpti_MetricValueKind valueKind;
        size_t valueKindSize = sizeof(valueKind);
        CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND,
                                       &valueKindSize, &valueKind));
        switch (valueKind) {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
                printf("%f", value.metricValueDouble);
            break;
            case CUPTI_METRIC_VALUE_KIND_UINT64:
                printf("%llu", (unsigned long long)value.metricValueUint64);
            break;
            case CUPTI_METRIC_VALUE_KIND_INT64:
                printf("%lld", (long long)value.metricValueInt64);
            break;
            case CUPTI_METRIC_VALUE_KIND_PERCENT:
                printf("%f%%", value.metricValuePercent);
            break;
            case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
                printf("%llu", (unsigned long long)value.metricValueThroughput);
            break;
    	    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
                printf("%u\n", (unsigned int)value.metricValueUtilizationLevel);
      	    break;
	    case CUPTI_METRIC_VALUE_KIND_FORCE_INT:
	    	printf("force int\n");
                printf("%d\n", (unsigned int)value.metricValueInt64);
	    break;
            default:
                fprintf(stderr, "error: unknown value kind\n");
            exit(-1);
    }
}

void print_metric_value(const char *metric_names, CUpti_MetricID metric_id, CUpti_MetricValue metric_value)
{
    printf("%s,", metric_names);
    print_metric(metric_id, metric_value);
    printf(" ");
    printf("\n");
}


MetricData_Mul_t init_md_mul(CUdevice device)
{
    MetricData_Mul_t metric_data_mul = {0};
    metric_data_mul.device = device;
    metric_data_mul.metric_names =  availabel_metrics(metric_data_mul.device, &metric_data_mul.metric_num);

    metric_data_mul.metric_ids = (CUpti_MetricID*)calloc(sizeof(CUpti_MetricID), metric_data_mul.metric_num);
    metric_data_mul.metric_values = (CUpti_MetricValue*)calloc(sizeof(CUpti_MetricValue), metric_data_mul.metric_num);
    metric_data_mul.current_pass = 0;
    return metric_data_mul;
}

void finish_md_mul(MetricData_Mul_t *metric_data_mul)
{
    free(metric_data_mul->metric_ids);
    free(metric_data_mul->metric_datas);
    for(int i = 0; i < metric_data_mul->metric_num; i++) {
	free(metric_data_mul->metric_names[i]);
    }
    free(metric_data_mul->metric_names);
    for(int i = 0; i < metric_data_mul->metric_passes; i++) {
	free(metric_data_mul->metric_datas[i].event_ids);
	free(metric_data_mul->metric_datas[i].event_values);
    }
}
    

void start(MetricData_Mul_t *metric_data_mul)
{
    CUpti_EventGroupSets *metric_pass_data;
    MetricData_t *metric_data;
    unsigned int metric_passes;


    // make sure activity is enabled before any CUDA API
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

    DRIVER_API_CALL(cuDeviceGet(&(metric_data_mul->device), 0));
    DRIVER_API_CALL(cuCtxCreate(&(metric_data_mul->context), 0, metric_data_mul->device));


    CUPTI_CALL(cuptiSubscribe(&(metric_data_mul->subscriber), (CUpti_CallbackFunc)getMetricValueCallback, metric_data_mul));
    CUPTI_CALL(cuptiEnableCallback(1, metric_data_mul->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_CALL(cuptiEnableCallback(1, metric_data_mul->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    //metric_ids = (CUpti_MetricID*)calloc(sizeof(CUpti_MetricID), metric_num);
    for(int i = 0; i < metric_data_mul->metric_num; i++) {
        CUPTI_CALL(cuptiMetricGetIdFromName(metric_data_mul->device, metric_data_mul->metric_names[i], &metric_data_mul->metric_ids[i]));
    }

    // get the number of passes required to collect all the events
    // needed for the metric and the event groups for each pass
    if(metric_data_mul->metric_num > 0) {
        CUPTI_CALL(cuptiMetricCreateEventGroupSets(metric_data_mul->context, sizeof(CUpti_MetricID) * metric_data_mul->metric_num, metric_data_mul->metric_ids, &metric_pass_data));
    }

    metric_passes = metric_pass_data->numSets;
    metric_data = (MetricData_t *)calloc(sizeof(MetricData_t), metric_passes);

    for(int i = 0; i < metric_passes; i++) {
        int total_events = 0;
        uint32_t num_events = 0;
        size_t num_events_size = sizeof(num_events);
        for(int j = 0; j < metric_pass_data->sets[i].numEventGroups; j++) {
            CUPTI_CALL(cuptiEventGroupGetAttribute(
                        metric_pass_data->sets[i].eventGroups[j],
                        CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                        &num_events_size, &num_events));
            total_events += num_events;
        }
        metric_data[i].event_groups = metric_pass_data->sets + i;
        metric_data[i].num_events = total_events;
	metric_data[i].event_ids = (CUpti_EventID *)calloc(sizeof(CUpti_EventID), total_events);
	metric_data[i].event_values = (uint64_t *)calloc(sizeof(uint64_t), total_events);
	
    }
  
    //metric_data_mul->current_ker_idx++;
    metric_data_mul->metric_passes = metric_passes;
    metric_data_mul->metric_datas = metric_data;
}


void end(MetricData_Mul_t *metric_data_mul)
{
	MetricData_t *metric_data = metric_data_mul->metric_datas;
	unsigned int metric_passes = metric_data_mul->metric_passes;
	int total_events = 0;
	for(int j = 0; j < metric_passes; j++) {
	    total_events += metric_data[j].num_events;
	}
	CUpti_MetricValue metric_value;
	CUpti_EventID *event_ids = (CUpti_EventID *)calloc(sizeof(CUpti_EventID), total_events);
	uint64_t *event_values = (uint64_t *)calloc(sizeof(uint64_t), total_events);

	int running_sum = 0;
        for(int j = 0; j < metric_passes; j++) {
	    for(int k = 0; k < metric_data[j].num_events; k++) {
		event_ids[running_sum + k] = metric_data[j].event_ids[k];
		event_values[running_sum + k] = metric_data[j].event_values[k];
	    }
	    running_sum += metric_data[j].num_events;
        }
	printf("kernel duration is: %lu\n", kernelDuration);
	for(int j = 0; j < metric_data_mul->metric_num; j++) {
    	    cuptiMetricGetValue(metric_data_mul->device, 
				 metric_data_mul->metric_ids[j],
                                 total_events * sizeof(CUpti_EventID),
                                 event_ids,
                                 total_events * sizeof(uint64_t),
                                 event_values,
                                 kernelDuration, &metric_value);
	    print_metric_value(metric_data_mul->metric_names[j], metric_data_mul->metric_ids[j], metric_value);
	    metric_data_mul->metric_values[j] = metric_value;
	}

	free(event_ids);
	free(event_values);

        CUPTI_CALL(cuptiEnableCallback(0, metric_data_mul->subscriber,
         	        CUPTI_CB_DOMAIN_RUNTIME_API,
                	CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
      	CUPTI_CALL(cuptiEnableCallback(0, metric_data_mul->subscriber,
        	        CUPTI_CB_DOMAIN_RUNTIME_API,
                	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
	CUPTI_CALL(cuptiUnsubscribe(metric_data_mul->subscriber));
}

	

