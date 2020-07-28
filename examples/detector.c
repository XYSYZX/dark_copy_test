#include "darknet.h"
#include "../src/bound_adjust.h"
#include<unistd.h>
#include<fcntl.h>
#include<dirent.h>
#include<sys/stat.h>
#include <time.h>


#define MAX_TEST_IMAGE 200
#define MAX_BOX 145
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

/*
** 图像检测网络训练函数（针对图像检测的网络训练）
** 输入： datacfg     训练数据描述信息文件路径及名称
**       cfgfile     神经网络结构配置文件路径及名称
**       weightfile  预训练参数文件路径及名称
**       gpus        GPU卡号集合（比如使用1块GPU，那么里面只含0元素，默认使用0卡号GPU；如果使用4块GPU，那么含有0,1,2,3四个元素；如果不使用GPU，那么为空指针）
**       ngpus       使用GPUS块数，使用一块GPU和不使用GPU时，nqpus都等于1
**       clear       
** 说明：关于预训练参数文件weightfile，
*/
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "backup");
    // 该文件可以由darknet提供的scripts/voc_label.py根据自行在网上下载的voc数据集生成，所以说是默认路径，其实也需要使用者自行调整，也可以任意命名，不一定要为train.list，
    srand(time(0));
    // 提取配置文件名称中的主要信息，用于输出打印，比如提取cfg/yolo.cfg中的yolo，用于下面的输出打印
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);    //shallow is 0
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 10000 && i%1000 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    for(i = 0; i < ngpus; ++i){
#ifdef GPU
        free_network(net);
#endif
    }

}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);  //make an empty image, in image.c

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);   //in data.c
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
    free_network(net);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .3;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
    free_network(net);
}

//void validate_detector_recall(char *cfgfile, char *weightfile)
void validate_detector_recall(char *datacfg,char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

//    list *plist = get_paths("data/coco_val_5k.list");
    list *plist = get_paths("/vlsilab/z-yang/dataset/coco/5k.new");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .3;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
    free_network(net);
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(const box_prob *)pa;
    box_prob b = *(const box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int *get_network_nweights(network *net)
{
	int *all_nweights = calloc(net->n, sizeof(int));
	for(int i = 0; i < net->n; ++i){
		if(net->layers[i].type == CONVOLUTIONAL)
			all_nweights[i] = net->layers[i].nweights;
	}
	return all_nweights;
}
		

float vulner_detector(char *datacfg, char *cfgfile, char *weightfile, float thresh, char *image_file, char *cost_file, int *bit_mode, int recall, int mAP)
{
    network *net = load_network(cfgfile, weightfile, 0); //load network
    set_batch_network(net, 1);
	layer l = net->layers[net->n-1];
 
 	list *plist = get_paths(image_file);
	char **paths = (char **)list_to_array(plist);
	char *path = paths[0];

	float nms = .2;
	float iou_thresh = .5;
	int i ,j ,k, m, n, x, y;

	FILE *cf = fopen(cost_file, "w");

    data val;
	load_args args = get_base_args(net);
	args.coords = l.coords;
    args.n = 1;
    args.m = 1;
	args.classes = l.classes;
    args.jitter = l.jitter;
    args.num_boxes = l.max_boxes;
    args.d = &val;
    args.type = DETECTION_DATA;
	args.threads = 1;
	args.paths = paths;

	pthread_t load_thread = load_data_in_thread(args);
	pthread_join(load_thread, 0);
	set_network_input_truth(net, val);

	int *all_nweights = get_network_nweights(net);

    char labelpath[4096];
	find_replace(path, "images", "labels", labelpath);
	find_replace(labelpath, "JPEGImages", "labels", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);

	int num_labels = 0;
	int nboxes;
    box_label *truth = read_boxes(labelpath, &num_labels);
	
	for(i = 0; i < net->n; i++){
		layer *lptr = &net->layers[i];
		for(j = 0; j < all_nweights[i]; j++){
			for(k = 0; k < 32; k++){
				if(bit_mode[k] == 1){
					inject_noise_weights_onebit(lptr, j, k);
					float *out = network_predict_single(net, val);
					save_cost(i, j, k, net->cost, cf);
					detection *dets;
					if(recall || mAP){
						dets = get_network_boxes(net, val.w, val.h, thresh, .5, 0, 1, &nboxes);
			        	if (nms) do_nms_obj(dets, nboxes, 1, nms);
					}
					if(recall){
						int total = 0;
						int correct = 0;
						int proposals = 0;
						float avg_iou = 0;

						for(m = 0; m < nboxes; m++){
							if(dets[m].objectness > thresh){
								++proposals;
							}
						}
						for(m = 0; m < num_labels; ++m) {
							++total;
							box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
							float best_iou = 0;
							for(n = 0; n < l.w*l.h*l.n; ++n){
								float iou = box_iou(dets[k].bbox, t);
								if(dets[k].objectness > thresh && iou > best_iou){
									best_iou = iou;
								}
							}
							avg_iou += best_iou;
							if(best_iou > iou_thresh){
								++correct;
							}
						}
						fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
					}
					/*
					if(mAP){
						
					}
					*/
				}
			}
		}
	}
	fclose(cf);
	free(path);
	free(paths);
	free_network(net);
}





float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, char *wbound_file, char *obound_file)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    if(net->dirty) inject_noise_weights(net);
    if(net->dirty_limit) inject_noise_weights_limit(net);
    
    if(net->limit_weight){
        load_weight_bound(net, wbound_file);
        check_weights(net);
    }
    if(net->limit_output){
        load_output_bound_8(net, obound_file);
    }
    //int initial_batch;
      
    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    const float thresh = .01;
    const float nms = .45;
    const int robust_dirty = 0;
    //const float iou_thresh = 0.5;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = calloc(nthreads, sizeof(image));
    image* val_resized = calloc(nthreads, sizeof(image));
    image* buf = calloc(nthreads, sizeof(image));
    image* buf_resized = calloc(nthreads, sizeof(image));
    pthread_t* thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net->w;
    args.h = net->h;
    //args.c = net->c;
    args.type = LETTERBOX_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;   //true positive?
    int fp_for_thresh = 0;

    box_prob* detections = calloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;  //所有样本中所有真实值的总和

    int* truth_classes_count = calloc(classes, sizeof(int));

    // For multi-class precision and recall computation
    float *avg_iou_per_class = calloc(classes, sizeof(float));
    int *tp_for_thresh_per_class = calloc(classes, sizeof(int));
    int *fp_for_thresh_per_class = calloc(classes, sizeof(int));

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "\r%d", i);
        for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && (i + t) < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            char *id = basecfg(path);
            float *X = val_resized[t].data;

            time_t tm = time(0); 
            if(net->dirty && robust_dirty){
                inject_noise_weights(net);
                printf("time for inject noise in weights: %ld\n", time(0) - tm);
            }
            tm = time(0);
            if(net->dirty_limit && robust_dirty){
                inject_noise_weights_limit(net);
                printf("time for inject noise in weights: %ld\n", time(0) - tm);
            }
            tm = time(0);
            if(net->limit_weight && net->robust){
                check_weights(net);
                printf("time for check weights: %ld\n", time(0) - tm);
            }
            network_predict(net, X);

            int nboxes = 0;
            //float hier_thresh = 0;
            int w = val[t].w;
            int h = val[t].h;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, 0, 1, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);

            char labelpath[4096];
            find_replace(path, "images", "labels", labelpath);
            find_replace(labelpath, "JPEGImages", "labels", labelpath);
            find_replace(labelpath, ".jpg", ".txt", labelpath);
            find_replace(labelpath, ".JPEG", ".txt", labelpath);

            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);
            int j;
            for (j = 0; j < num_labels; ++j) {
                truth_classes_count[truth[j].id]++;   //每张图片中真实存在的物体（txt文件）
            }

            // difficult
            /*
            box_label *truth_dif = NULL;
            int num_labels_dif = 0;
            if (paths_dif)
            {
                char *path_dif = paths_dif[image_index];

                char labelpath_dif[4096];
                replace_image_to_label(path_dif, labelpath_dif);

                truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
            }
            */
            const int checkpoint_detections_count = detections_count;

            int i;
            for (i = 0; i < nboxes; ++i) {

                int class_id;
                for (class_id = 0; class_id < classes; ++class_id) {
                    float prob = dets[i].prob[class_id];
                    if (prob > 0) {
                        detections_count++;
                        detections = (box_prob*)realloc(detections, detections_count * sizeof(box_prob));
                        detections[detections_count - 1].b = dets[i].bbox;
                        detections[detections_count - 1].p = prob;
                        detections[detections_count - 1].image_index = image_index;
                        detections[detections_count - 1].class_id = class_id;
                        detections[detections_count - 1].truth_flag = 0;
                        detections[detections_count - 1].unique_truth_index = -1;

                        int truth_index = -1;
                        float max_iou = 0;  //针对每个box里面每个prob匹配样本文件，计算max_iou
                        for (j = 0; j < num_labels; ++j)
                        {
                            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                            float current_iou = box_iou(dets[i].bbox, t);
                            if (current_iou > iou_thresh && class_id == truth[j].id) {
                                if (current_iou > max_iou) {
                                    max_iou = current_iou;
                                    truth_index = unique_truth_count + j;
                                }
                            }
                        }

                        // best IoU
                        if (truth_index > -1) {   //该box的该prob成功匹配样本
                            detections[detections_count - 1].truth_flag = 1;
                            detections[detections_count - 1].unique_truth_index = truth_index;
                        }
                       
                        // calc avg IoU, true-positives, false-positives for required Threshold
                        if (prob > thresh_calc_avg_iou) {
                            int z, found = 0;
                            //回顾之前的预测框, 要是之前的预测框中存在已经和本框对应的样本匹配成功的框，本框就不算数拉
                            for (z = checkpoint_detections_count; z < detections_count - 1; ++z) {
                                if (detections[z].unique_truth_index == truth_index) {
                                    found = 1; break;
                                }
                            }

                            if (truth_index > -1 && found == 0) {
                                avg_iou += max_iou;
                                ++tp_for_thresh;
                                avg_iou_per_class[class_id] += max_iou;
                                tp_for_thresh_per_class[class_id]++;
                            }
                            else{
                                fp_for_thresh++;
                                fp_for_thresh_per_class[class_id]++;
                            }
                        }
                    }
                }
            }

            unique_truth_count += num_labels;


            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }


    if ((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

    int class_id;
    for(class_id = 0; class_id < classes; class_id++){
        if ((tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]) > 0)
            avg_iou_per_class[class_id] = avg_iou_per_class[class_id] / (tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]);
    }

    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t** pr = (pr_t**)calloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t*)calloc(detections_count, sizeof(pr_t));
    }
    printf("\n detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int* detection_per_class_count = (int*)calloc(classes, sizeof(int));
    for (j = 0; j < detections_count; ++j) {
        detection_per_class_count[detections[j].class_id]++;
    }

    int* truth_flags = (int*)calloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else
                pr[d.class_id][rank].fp++;
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                    printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, detection_per_class_count[i], tp+fp, tp, fp);
            }
        }
    }

    free(truth_flags);


    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;

        // MS COCO - uses 101-Recall-points on PR-chart.
        // PascalVOC2007 - uses 11-Recall-points on PR-chart.
        // PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
        // ImageNet - uses Area-Under-Curve on PR-chart.

        // correct mAP calculation: ImageNet, PascalVOC 2010-2012
        if (map_points == 0)
        {
            double last_recall = pr[i][detections_count - 1].recall;
            double last_precision = pr[i][detections_count - 1].precision;
            for (rank = detections_count - 2; rank >= 0; --rank)
            {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) {
                    last_precision = pr[i][rank].precision;
                }

                avg_precision += delta_recall * last_precision;
            }
        }
        // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        else
        {
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points-1);
                double cur_precision = 0;
                for (rank = 0; rank < detections_count; ++rank)
                {
                    if (pr[i][rank].recall >= cur_recall) {    // > or >=
                        if (pr[i][rank].precision > cur_precision) {
                            cur_precision = pr[i][rank].precision;
                        }
                    }
                }
                //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        printf("class_id = %d, ap = %2.2f%%   \t (TP = %d, FP = %d) \n",
            i, avg_precision * 100, tp_for_thresh_per_class[i], fp_for_thresh_per_class[i]);

        //float class_precision = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
        //float class_recall = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
        //printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n", class_precision, class_recall, avg_iou_per_class[i]);
        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
    if (map_points) printf("used %d Recall-points \n", map_points);
    else printf("used Area-Under-Curve for each unique Recall \n");

    printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh, mean_average_precision, mean_average_precision * 100);

    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);

    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);

    fprintf(stderr, "Total Detection Time: %d Seconds\n", (int)(time(0) - start));
    printf("\nSet -points flag:\n");
    printf(" `-points 101` for MS COCO \n");
    printf(" `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) \n");
    printf(" `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset\n");

    // free memory
    //free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    free_network(net);
    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    return mean_average_precision;
}


/** 本函数是检测模型的一个前向推理测试函数.
 * @param datacfg       数据集信息文件路径（也即cfg/*.data文件），文件中包含有关数据集的信息，比如cfg/coco.data
 * @param cfgfile       网络配置文件路径（也即cfg/*.cfg文件），包含一个网络所有的结构参数，比如cfg/yolo.cfg
 * @param weightfile    已经训练好的网络权重文件路径，比如darknet网站上下载的yolo.weights文件
 * @param filename      待进行检测的图片路径（单张图片）
 * @param thresh        阈值，类别检测概率大于该阈值才认为其检测结果有效
 * @param fullscreen
*/
 //detect images in bunch
void test_detector(char *datacfg, char *cfgfile, char *weightfile,  float thresh, float hier_thresh, char *rf_name, char *imf_name, int max, int max_box)
{
    double time_parser=what_time_is_it_now();
    
    network *net = load_network(cfgfile, weightfile, 0); //load network
    printf("Parse cfg, weight in %f seconds. \n", what_time_is_it_now()-time_parser);
    set_batch_network(net, 1);            //set value of every batch as 1,because inference
    srand(2222222);
    double time;
    double time_detection;
    
    float nms=.2;

    FILE *rf;  //create a file to save output(right_file)
    FILE *imf;
 
    rf = fopen(rf_name, "w"); //write detect result to rightfile
    imf = fopen(imf_name, "r");  //open test image file, to read imagepath
    
    int img_num=0;

    if(rf==NULL||imf==NULL)
    {
	    printf("cannot open right file or input image file!\n");
	    return;
    }

    while(!feof(imf)){   //until last line of image file
        char imagepath[256];
        fgets(imagepath, 256, imf);  //get path of an image in imagelist
        char *find = strchr(imagepath, '\n');
        if(find) *find = '\0';
        printf("the imagepath name is: %s*\n", imagepath);

	    image im = load_image_color(imagepath, 0, 0);
        image sized = letterbox_image(im, net->w, net->h); 
        layer l = net->layers[net->n-1];

        float *X = sized.data;  //data of an image

        time=what_time_is_it_now();
        network_predict(net, X);  //return output of network, only forward_network
        printf("%s: Predicted in %f seconds.\n", imagepath, what_time_is_it_now()-time);
            
        int nboxes = 0;
        time_detection=what_time_is_it_now();
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        printf("there are %d boxes in the dets\n", nboxes);

        if (nms) 
            do_nms_sort(dets, nboxes, l.classes, nms);  //in box.c
        
        save_rightline(img_num, dets, nboxes, l.classes, max_box, rf);

        free_detections(dets, nboxes);
	    printf("detection by write in %f seconds.\n", what_time_is_it_now()-time_detection);
	
        free_image(im);
        free_image(sized);
	    img_num++;

	    if(img_num >= max) break; 		    
    }
    fclose(rf);
    fclose(imf);
    free_network(net);
} 
   
/*
void write_head(FILE *fp)
{
	time_t seconds;
	time(&seconds);
	char l[256];
	sprintf(l, "now it is %s, the size of BLOCK is: %d", ctime(&seconds), BLOCK);
	fputs(l, fp);
}
*/

void compress(char *filename)
{
    char sys[256]={'\0'};
    sprintf(sys, "gzip %s ", filename);
    printf("%s\n", sys);
    system(sys);
}

void compare_detector(char *datacfg, char *cfgfile, char *weightfile,  float thresh, float hier_thresh, char * rf_name, char *imf_name, char *wf_path, char *af_path, int max_img, float df, int max_box, char *wbound_file, char *obound_file)
{
    double time_parser=what_time_is_it_now();
    network *net = load_network(cfgfile, weightfile, 0); //load network
    fprintf(stderr, "Parse cfg, weight in %f seconds. \n", what_time_is_it_now()-time_parser);
    set_batch_network(net, 1);
    srand(2222222);

    if(net->limit_weight){
        load_weight_bound(net, wbound_file);
        check_weights(net);
    }
    if(net->limit_output){
	fprintf(stderr, "lets check output!\n");
        load_output_bound(net, obound_file);
    }

    double times;
    float nms=.2;

    FILE *rf;  //create a file to save output(network_output)
    FILE *imf;  //input image list

    rf = fopen(rf_name, "r"); //open pretested right file
    imf = fopen(imf_name, "r");  //open test image file

    if(rf==NULL||imf==NULL)
    {
	    printf("cannot open right file or input image file!\n");
	    return;
    }
    int img_num=0;   //num of tested image, for recording which image has SDC 
    int iter=0;
    while(!feof(rf) && img_num < max_img)
    {	times = what_time_is_it_now();
        FILE *fp, *wf;
        char wrong_name[256], all_name[256];
        sprintf(wrong_name, "%s%lf", wf_path, times);
        sprintf(all_name, "%s%lf", af_path, times);
        fp = fopen(all_name, "w"); 
        wf = fopen(wrong_name, "w");
        if(!fp || !wf){
            printf("can not open wf or fp!\n");
            return;
        }
        iter=0;

        char imagepath[256];
        fgets(imagepath, 256, imf);  //get path of an image in imagelist
        char *find = strchr(imagepath, '\n');
        if(find) *find = '\0';
        //printf("the imagepath name is: %s*\n", imagepath);

        gold_ans *gd = load_rightline(rf);
    
        image im = load_image_color(imagepath, 0, 0);
        image sized = letterbox_image(im, net->w, net->h); 
        layer l = net->layers[net->n-1];

        float *X = sized.data;  //data of an image  //care where the point is!!!! important in fp16!!! ok, data is calloced
        //printf("prepare in %f seconds.\n", what_time_is_it_now()-times);
        while(iter<100){   //start 200 iterations
            times = what_time_is_it_now();

            if(net->limit_weight){
                check_weights(net);
                //printf("check weights in %f seconds.\n", what_time_is_it_now()-times);
            }
            network_predict(net, X);  //return output of network, only forward_network
            //if(net->limit_output) check_outputs(net);
            //printf("%s: Predicted in %f seconds.\n", imagepath, what_time_is_it_now()-times);
                
            int nboxes = 0;
            detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
            if (nms) 
                do_nms_sort(dets, nboxes, l.classes, nms);  //in box.c
            //times = what_time_is_it_now();
            compare_detections(img_num, dets, nboxes, l.classes ,wf, fp, df, gd);  //can be functioned on cuda? or multi thread
            free_detections(dets, nboxes);
            //printf("cpu in %f seconds.\n", what_time_is_it_now()-times);
            iter++;   //num of iterations
        }
            
        free_image(im);
        free_image(sized); 
        free_gold_ans(gd);

        img_num++; 
        if(img_num >= max_img){
            fseek(rf, 0, SEEK_SET);
    	    fseek(imf, 0, SEEK_SET);	    
            img_num=0;
        }
        
        fclose(fp);
        fclose(wf);
        compress(all_name);
        compress(wrong_name);
        
    }
    fclose(rf);
    fclose(imf);
    free_network(net);
} 

void get_detector_weight_bound(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    //list *options = read_data_cfg(datacfg);
    network *net = load_network(cfgfile, weightfile, 0);
    if(net->weight_bound){
        get_weight_bound_network(net);
        save_weight_bound(net, outfile);
    }
    free_network(net);
}

void get_detector_output_bound(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    list *options = read_data_cfg(datacfg);    
    char *train_images = option_find_str(options, "train", "data/train.list");

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);


    int m = (int)plist->size * 0.2;
    int i=0;
    int t;
    //int *map = 0;

    //float thresh = .3;
    //float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            float *X = val_resized[t].data;
            network_predict(net, X);
            if(net->output_bound){
                get_output_bound_network(net);
            }
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    save_output_bound(net, outfile);
    fprintf(stderr, "Total Output Bound Time: %f Seconds\n", what_time_is_it_now() - start);
    free_network(net);
}

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);  //in utils.c
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;  //calculate the number of gpu
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;  //return the location of ',', store the id num of every gpu in gpus[]
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    int map_points = find_int_arg(argc, argv, "-points", 0);
    //int noise_freq = find_int_arg(argc, argv, "-noise_freq", 1);
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    //float thresh = find_float_arg(argc, argv, "thresh", .5);
    int letter_box = find_arg(argc, argv, "-letter_box");
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *rf_name = find_char_arg(argc, argv, "-rf_name", NULL);
    char *imf_name = find_char_arg(argc, argv, "-imf_name", NULL);
    char *wf_path = find_char_arg(argc, argv, "-wf_name", NULL);
    char *af_path = find_char_arg(argc, argv, "-af_name", NULL);

    char *wbound_file = find_char_arg(argc, argv, "-wbound_file", NULL);
    char *obound_file = find_char_arg(argc, argv, "-obound_file", NULL);
	char *cost_file = find_char_arg(argc, argv, "-cost_file", NULL);
	char *image_file = find_char_arg(argc, argv, "-image_file", NULL);
    
    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    //char *filename = (argc > 6) ? argv[6]: 0;
    char *filename = find_char_arg(argc, argv, "-input", 0);

    int max_box = 145;
    int max_img = 200;
    float df = 0.00001;
	int recall = 0;
	int mAP = 0;
	int bit_mode[32] = {0, 0, 0, 0, 0, 0, 0, 0, 
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0}; 

    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, thresh, hier_thresh, rf_name, imf_name, max_img, max_box);
    if(0==strcmp(argv[2], "compare")) compare_detector(datacfg, cfg, weights, thresh, hier_thresh, rf_name, imf_name, wf_path, af_path, max_img, df, max_box, wbound_file, obound_file);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
    else if(0 == strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, wbound_file, obound_file);
    else if(0==strcmp(argv[2], "weight_bound")) get_detector_weight_bound(datacfg, cfg, weights, wbound_file);
    else if(0==strcmp(argv[2], "output_bound")) get_detector_output_bound(datacfg, cfg, weights, obound_file);
    else if(0==strcmp(argv[2], "vulnerable")) vulner_detector(datacfg, cfg, weights, thresh, image_file, cost_file, bit_mode, recall, mAP);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}

