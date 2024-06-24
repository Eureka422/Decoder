#include <math.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <zbar.h>
#include <pthread.h>
#include <semaphore.h>
#include <arpa/inet.h>
#include <string.h>
#include <atomic>

using namespace std;

/*
 * This file is used to decode QRcode
 * Four threads are created, occupying 4 cores, respectively
 *     0: grab image from camera, using OpenCV
 *     1: process the image on the GPU, using OpenCL
 *     2: decode the even frame, using zbar
 *     3: decode the odd frame, using zbar
*/

#define CAM_WIDTH 800
#define CAM_HEIGHT 600
#define PI 3.14159265358979
#define rad2deg 180/PI

// #define IP_ADDR "192.168.149.34"
#define PORT 8080
//800x600
// camera matrix
double fx = 525.978756527803;
double cx = 401.280196577630;
double fy = 525.914523890096;
double cy = 308.309782426420;
double k1 = -0.0427714552282332;
double k2 = -0.0520163509703581;
double p1 = 0.000211157277284847;
double p2 = -0.0000458493815429687;
double k3 = 0.0374695129349998;
cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) <<
                        fx, 0.0, cx,
                        0.0, fy, cy,
                        0.0, 0.0, 1.0);
cv::Mat dist_coeffs = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);
// QRcode size
double marker_width = 3.5; //unit  cm
double marker_height = 3.5; //unit  cm

// ==================global variables===================
atomic<int> block_size(0); // store the block size of the QRcode
int even_cut_flag_g = 0; // 1:find the code & cut success 0: cant find code ,cut fail
int odd_cut_flag_g = 0;  // 1:find the code & cut success 0: cant find code ,cut fail
int border0_g[4] = {CAM_WIDTH, 0, CAM_HEIGHT, 0};  // left right top bottom
int border1_g[4] = {CAM_WIDTH, 0, CAM_HEIGHT, 0};  // left right top bottom
// openCL
cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel;
// socket
int server_fd, client_fd;

// ==================global semaphore===================
sem_t gray_capped_sem;
sem_t gray_cloned_sem;
sem_t even_decode_sem;
sem_t odd_decode_sem; 
sem_t even_cutted_sem;    
sem_t odd_cutted_sem;    
// ==================global mutex======================
pthread_mutex_t socket_mutex;

// ====================global image=====================
// func_grab -> func_gpu
cv::Mat capped_img_g(cv::Size(CAM_WIDTH, CAM_HEIGHT), CV_8UC1); // store the captured image (grayscale)
// func_gpu -> func_decode0
cv::Mat even_cut_img_g; // Size cannot be determined
cv::Mat odd_cut_img_g; // Size cannot be determined
cv::Mat even_samp_img_g(cv::Size(CAM_WIDTH/2, CAM_HEIGHT/2), CV_8UC1);
cv::Mat odd_samp_img_g(cv::Size(CAM_WIDTH/2, CAM_HEIGHT/2), CV_8UC1);
// func_decode0 -> func_decode1

struct XClDeviceInfo
{
	cl_device_id id;
	cl_device_type device_type;
	cl_uint vendor_id;
	cl_uint workitem_max_dim;
	size_t max_workgroup_size;
	cl_ulong max_mem_size;
};

struct XClPlatformInfo
{
	cl_platform_id id;
	std::string profile;
	std::string version;
	std::string name;
	std::string vendor;
	std::string extentsion;
	std::vector<XClDeviceInfo> devices;
};

std::vector<XClPlatformInfo> g_platforms;

// ====================func declaration=================
int set_camera(cv::VideoCapture cap);
int32_t QueryDevices(cl_platform_id platform_id, std::vector<XClDeviceInfo> &out_dev_list);
int32_t QueryPlatforms();
int32_t LoadClFileToString(const string &cl_file_path, string &out_string);
void calc_r_t(vector<vector<cv::Point2f>> corners, cv::Size board_size, cv::Mat camera_matrix, cv::Mat dist_coeffs, vector<cv::Mat>& rvecs, vector<cv::Mat>& tvecs);


// ====================func definition==================
/*
 * Function: func_decode0
 * Description: Decode the cutted image or sampled image
 * Input: None
 * Output: None
*/
void* func_decode0(void* arg)
{
    int frame_count = 0;
    int cut_success = 0;
    struct timeval t_cur, t_last;
    cv::Mat gray_img;

    zbar::Image::SymbolIterator symbol;
    zbar::ImageScanner scanner; // create a scanner
    scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);  // enable the scanner

    double message[8];  // socket message to send

    gettimeofday(&t_last, NULL);
    while (1) {
        sem_wait(&even_cutted_sem);
        if (even_cut_flag_g == 1) {
            gray_img = even_cut_img_g.clone();
            cut_success = 1;
        } else {
            gray_img = even_samp_img_g.clone();
            cut_success = 0;
        }
        sem_post(&even_decode_sem);

        // decode
        zbar::Image image(gray_img.cols, gray_img.rows, "Y800", gray_img.data, gray_img.cols * gray_img.rows);
        scanner.scan(image);    // scan the image
        symbol = image.symbol_begin();
        // TODO: process the decode result
        // get QRcode corner
        vector<vector<cv::Point2f>> corners;
        for (; symbol != image.symbol_end(); ++symbol) {
            vector<cv::Point2f> corner;
            if (cut_success) {
                for (int i = 0; i < symbol->get_location_size(); i++) {
                    corner.push_back(cv::Point2f(symbol->get_location_x(i) + border0_g[0], symbol->get_location_y(i) + border0_g[2]));
                }
            } else {
                for (int i = 0; i < symbol->get_location_size(); i++) {
                    corner.push_back(cv::Point2f(symbol->get_location_x(i) * 2, symbol->get_location_y(i) * 2));
                }
            }
            corners.push_back(corner);
            std::cout << "decoded " << symbol->get_type_name() << " symbol \"" << symbol->get_data() << '"' << " " << corners.size() << std::endl;
        }

        // calculate rvecs & tvecs
        vector<cv::Mat> rvecs, tvecs;
        calc_r_t(corners, cv::Size(3, 3), camera_matrix, dist_coeffs, rvecs, tvecs);

        // calculate the frame rate
        frame_count++;
        if (frame_count >= 120) {
            frame_count = 0;
            gettimeofday(&t_cur, NULL);
            double time = (t_cur.tv_sec - t_last.tv_sec) + (t_cur.tv_usec - t_last.tv_usec) / 1000000.0;
            double fps = 120 / time;
            std::cout << "even image decode FPS: " << fps << std::endl;
            gettimeofday(&t_last, NULL);
        }

        // send message to client
        if (rvecs.size() > 0) {
            message[0] = tvecs[0].at<double>(0, 0); // x
            message[1] = tvecs[0].at<double>(1, 0); // y
            message[2] = tvecs[0].at<double>(2, 0); // z
            message[3] = rvecs[0].at<double>(0, 0); // x
            message[4] = rvecs[0].at<double>(1, 0); // y
            message[5] = rvecs[0].at<double>(2, 0); // z
            pthread_mutex_lock(&socket_mutex);  // lock
            send(client_fd, message, sizeof(message), 0);
            pthread_mutex_unlock(&socket_mutex); // unlock
        }
    }
    return ((void *)0);
}

/*
 * Function: func_decode1
 * Description: Decode the cutted image or sample image
 * Input: None
 * Output: None
*/
void* func_decode1(void* arg)
{
    int frame_count = 0;
    int cut_success = 0;
    struct timeval t_cur, t_last;
    cv::Mat gray_img;

    zbar::ImageScanner scanner; // create a scanner
    scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);  // enable the scanner
    zbar::Image::SymbolIterator symbol;

    double message[8];  // socket message to send

    gettimeofday(&t_last, NULL);
    while (1) {
        sem_wait(&odd_cutted_sem);
        if (odd_cut_flag_g == 1) {
            gray_img = odd_cut_img_g.clone();
            cut_success = 1;
        } else {
            gray_img = odd_samp_img_g.clone();
            cut_success = 0;
        }
        sem_post(&odd_decode_sem);

        // decode
        zbar::Image image(gray_img.cols, gray_img.rows, "Y800", gray_img.data, gray_img.cols * gray_img.rows);
        scanner.scan(image);    // scan the image
        symbol = image.symbol_begin();
        // todo: process the decode result
        // get QRcode corner
        vector<vector<cv::Point2f>> corners;
        for (; symbol != image.symbol_end(); ++symbol) {
            vector<cv::Point2f> corner;
            // put 0(left up), 1(left down), 2(right down), 3(right up) to the corners in order
            if (cut_success) {
                for (int i = 0; i < symbol->get_location_size(); i++) {
                    corner.push_back(cv::Point2f(symbol->get_location_x(i) + border1_g[0], symbol->get_location_y(i) + border1_g[2]));
                }
            } else {
                for (int i = 0; i < symbol->get_location_size(); i++) {
                    corner.push_back(cv::Point2f(symbol->get_location_x(i) * 2, symbol->get_location_y(i) * 2));
                }
            }
            corners.push_back(corner);
        }

        // calculate block_size
        if (corners.size() > 0) {
            double d1 = sqrt(pow(corners[0][0].x - corners[0][1].x, 2) + pow(corners[0][0].y - corners[0][1].y, 2));
            double d2 = sqrt(pow(corners[0][1].x - corners[0][2].x, 2) + pow(corners[0][1].y - corners[0][2].y, 2));
            // atomic operation
            block_size.store((int)(d1 + d2) / 2 / 21); // QRcode has 21x21 blocks
        }

        // calculate rvec & tvec
        vector<cv::Mat> rvecs, tvecs;
        calc_r_t(corners, cv::Size(marker_height, marker_width), camera_matrix, dist_coeffs, rvecs, tvecs);

        // calculate the frame rate
        frame_count++;
        if (frame_count >= 120) {
            frame_count = 0;
            gettimeofday(&t_cur, NULL);
            double time = (t_cur.tv_sec - t_last.tv_sec) + (t_cur.tv_usec - t_last.tv_usec) / 1000000.0;
            double fps = 120 / time;
            std::cout << "odd image decode1 FPS: " << fps << std::endl;
            gettimeofday(&t_last, NULL);
        }

        // send message to the client:tvecs, fps
        if (rvecs.size() > 0) {
            message[0] = tvecs[0].at<double>(0, 0); // x
            message[1] = tvecs[0].at<double>(1, 0); // y
            message[2] = tvecs[0].at<double>(2, 0); // z
            message[3] = rvecs[0].at<double>(0, 0); // x
            message[4] = rvecs[0].at<double>(1, 0); // y
            message[5] = rvecs[0].at<double>(2, 0); // z
            pthread_mutex_lock(&socket_mutex);  // lock
            send(client_fd, message, sizeof(message), 0);
            pthread_mutex_unlock(&socket_mutex); // unlock
        }
    }
    return ((void *)0);
}

/*
 * Function: func_grab
 * Description: Grab the image from the camera
 * Input: None
 * Output: None
*/
void* func_grab(void* arg)
{
    int frame_count = 0;
    char camera_select = 0; // 0:down 1:up
    struct timeval t_cur, t_last;
    int ret = 0;
    cv::Mat frame(cv::Size(CAM_WIDTH, CAM_HEIGHT), CV_8UC3);    // Create a frame to store the image
    // cv::Mat gray(cv::Size(CAM_WIDTH, CAM_HEIGHT), CV_8UC1);     // Create a frame to store the grayscale image
    cv::VideoCapture cap_down(0);   // ov9281
    cv::VideoCapture cap_up(9); // usb camera
    // set camera
    ret = set_camera(cap_down);
    if (ret < 0) {
        std::cerr << "Error: Set the camera_down failed(" << ret << ")" << std::endl;
    }
    ret = set_camera(cap_up);
    if (ret < 0) {
        std::cerr << "Error: Set the camera_up failed(" << ret << ")" << std::endl;
    }

    if (!cap_down.isOpened()) {
        std::cerr << "Error: Cannot open the camera_down" << std::endl;
        return ((void *)-1);
    } else {
        std::cout << "Success: Open the camera_down" << std::endl;
    }
    if (!cap_up.isOpened()) {
        std::cerr << "Error: Cannot open the camera_up" << std::endl;
        return ((void *)-1);
    } else {
        std::cout << "Success: Open the camera_up" << std::endl;
    }

    gettimeofday(&t_last, NULL);
    while (1) {
        if (camera_select == 0) {
            cap_down >> frame;
        } else {
            cap_up >> frame;
        }

        sem_wait(&gray_cloned_sem);
        cv::cvtColor(frame, capped_img_g, cv::COLOR_BGR2GRAY);    // convert the frame to grayscale
        sem_post(&gray_capped_sem);
        camera_select = cv::waitKey(1);

    }
    return ((void *)0);
}

/*
 * Function: func_gpu
 * Description: Process the image on the GPU，cut the image or sample the image
 * Input: None
 * Output: None
*/
void* func_gpu(void* arg)
{
    cv::Mat gray_img(cv::Size(CAM_WIDTH, CAM_HEIGHT), CV_8UC1);
    cv::Mat samp_img(cv::Size(CAM_WIDTH/2, CAM_HEIGHT/2), CV_8UC1, cv::Scalar(0)); 
    int img_index = 0;  // use to distinguish the even and odd image
    int cut_flag = 0;   // use to distinguish the cut success or fail

    QueryPlatforms();
    cl_int status = CL_SUCCESS;
    cl_device_id device_id = g_platforms[0].devices[0].id;
    // use the first platform and the first device, just follow the tutorial
    // 1. Create the context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Create the context failed" << std::endl;
        return ((void *)-1);
    }
    // 2. Create the command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &status);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Create the command queue failed" << std::endl;
        return ((void *)-1);
    }
    // 3. Create the program
    string cl_file_path = "../kernelfunc.cl";
    string cl_string_c;
    LoadClFileToString(cl_file_path, cl_string_c);
    if (cl_string_c.size() <= 0) {
        std::cerr << "Error: Create the cl_string failed" << std::endl;
    }
    const char* source_string = cl_string_c.c_str();
    size_t source_size[] = {strlen(source_string)};
    program = clCreateProgramWithSource(context, 1, &source_string, source_size, &status);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Create the program failed" << std::endl;
        return ((void *)-1);
    }
    if (program == nullptr) {
        std::cerr << "Error: Create the program1 failed" << std::endl;
        return ((void *)-1);
    }
    // 4. Build the program
    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Build the program failed" << std::endl;
        return ((void *)-1);
    }
    // 5. Create the kernel
    kernel = clCreateKernel(program, "image_filter", &status);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Create the kernel failed" << std::endl;
        return ((void *)-1);
    }
    cl_int pxl_bytes = 1;
    cl_int linebytes = CAM_WIDTH * pxl_bytes;   // used as kernerlfunc arguments

    while (1) {
        int border[4] = {CAM_WIDTH, 0, CAM_HEIGHT, 0};  // left right top bottom, This initialization is for iteration, in kernelfunc

        sem_wait(&gray_capped_sem);
        gray_img = capped_img_g.clone();
        sem_post(&gray_cloned_sem);

        cl_float block = block_size.load();  // get the block size

        // create mem, gray_img_mem, samp_img_mem, border_mem, used as kernelfunc arguments
        cl_mem gray_img_mem = clCreateBuffer(context, 
                                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                            CAM_WIDTH*CAM_HEIGHT*sizeof(cl_uchar), 
                                            (void*)gray_img.data, 
                                            &status);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Create the gray_img_mem failed" << std::endl;
            return ((void *)-1);
        }
        cl_mem samp_img_mem = clCreateBuffer(context, 
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                            CAM_WIDTH/2*CAM_HEIGHT/2*sizeof(cl_uchar), 
                                            (void*)samp_img.data, 
                                            &status);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Create the samp_img_mem failed" << std::endl;
            return ((void *)-1);
        }
        cl_mem border_mem = clCreateBuffer(context, 
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                        4*sizeof(int), 
                                        (void*)border, 
                                        &status);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Create the border_mem failed" << std::endl;
            return ((void *)-1);
        }
        // set arguments
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&gray_img_mem);  // set the argument 0, input image
        status = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&pxl_bytes);     // set the argument 1, pixel bytes
        status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&linebytes);     // set the argument 2, image width
        status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&border_mem);    // set the argument 3. border
        status = clSetKernelArg(kernel, 4, sizeof(cl_float), (void*)&block);       // set the argument 4, Qrcode block sizes
        status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&samp_img_mem);  // set the argument 5, output image


        size_t global_work_size[2] = {(size_t)CAM_WIDTH / 2, (size_t)CAM_HEIGHT / 2};   // set the global work size, left up quad image
        size_t local_work_size[2] = {1, 1};                             // set the local work size, single pixel

        // enqueue the kernel
        cl_event event = NULL;
        status = clEnqueueNDRangeKernel(command_queue, 
                                        kernel, 
                                        2, 
                                        NULL, 
                                        global_work_size, 
                                        local_work_size, 
                                        0, 
                                        NULL, &event);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Enqueue the kernel failed" << std::endl;
            return ((void *)-1);
        }

        // wait for the kernel to finish
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
        // take the sample image and border from the GPU
        status = clEnqueueReadBuffer(command_queue, 
                                    samp_img_mem, 
                                    CL_TRUE, 
                                    0, 
                                    (size_t)CAM_WIDTH/2*CAM_HEIGHT/2*sizeof(cl_uchar), 
                                    (void*)samp_img.data, 
                                    0, 
                                    NULL, 
                                    NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Read the buffer1 failed" << std::endl;
            return ((void *)-1);
        }
        status = clEnqueueReadBuffer(command_queue, 
                                    border_mem,
                                    CL_TRUE,
                                    0,
                                    (size_t)4*sizeof(int),
                                    (void*)border,
                                    0,
                                    NULL,
                                    NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Read the buffer2 failed" << std::endl;
            return ((void *)-1);
        }
        if (border[0] >= border[1] || border[2] >= border[3]) {   
            cut_flag = 0;
        } else {    // find the QRcode
            cut_flag = 1;
        }
        switch (img_index) {
            case 0: // even image
                sem_wait(&even_decode_sem);
                if (cut_flag == 1) {
                    border0_g[0] = border[0];
                    border0_g[1] = border[1];
                    border0_g[2] = border[2];
                    border0_g[3] = border[3];
                    even_cut_img_g = gray_img(cv::Range(border0_g[2], border0_g[3]), cv::Range(border0_g[0], border0_g[1]));
                    even_cut_flag_g = 1;
                } else {
                    even_samp_img_g = samp_img.clone();
                    even_cut_flag_g = 0;
                }
                sem_post(&even_cutted_sem);
                break;
            case 1: // odd image
                sem_wait(&odd_decode_sem);
                if (cut_flag == 1) {    // cut success
                    border1_g[0] = border[0];
                    border1_g[1] = border[1];
                    border1_g[2] = border[2];
                    border1_g[3] = border[3];
                    odd_cut_img_g = gray_img(cv::Range(border1_g[2], border1_g[3]), cv::Range(border1_g[0], border1_g[1]));
                    odd_cut_flag_g = 1;
                } else {                // cut fail
                    odd_samp_img_g = samp_img.clone();
                    odd_cut_flag_g = 0;
                }
                sem_post(&odd_cutted_sem);
                break;
            default:
                break;
        }
        img_index = (img_index + 1) % 2;


        clReleaseMemObject(gray_img_mem);
        clReleaseMemObject(samp_img_mem);
        clReleaseMemObject(border_mem);

    }

    return ((void *)0);
}

int main(void)
{
    // init semaphore
    sem_init(&gray_cloned_sem, 0, 1);   // <name>, <in-process>, <initial value>
    sem_init(&gray_capped_sem, 0, 0);
    sem_init(&even_cutted_sem, 0, 0);
    sem_init(&even_decode_sem, 0, 1);
    sem_init(&odd_cutted_sem,  0, 0);
    sem_init(&odd_decode_sem,  0, 1);

    pthread_mutex_init(&socket_mutex, NULL);

    // create a server
    struct sockaddr_in server_addr;
    int addrlen = sizeof(server_addr);
    server_fd = socket(AF_INET, SOCK_STREAM, 0);    // create a socket
    if (server_fd < 0) {
        std::cerr << "Error: Create the socket failed" << std::endl;
        return -1;
    }
    server_addr.sin_family = AF_INET;   // set the server address
    server_addr.sin_port = htons(PORT); // set the server port
    server_addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error: Bind the socket failed" << std::endl;
        return -1;
    }
    listen(server_fd, 5);   // listen the connection
    cout << "Server is listening" << endl;
    client_fd = accept(server_fd, (struct sockaddr *)&server_addr, (socklen_t*)&addrlen);
    if (client_fd < 0) {
        cout << "accept fail" << endl;
    } else {
        cout << "accept success" << endl;
    }

    pthread_t thread[4];    
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    // bind the function 'func_grab' to cpu0
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);
    pthread_create(&thread[0], &attr, func_grab, NULL);    // create a thread to grab the image from the camera

    // bind the function 'func_gpu' to cpu1
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);
    pthread_create(&thread[1], &attr, func_gpu, NULL);    // create a thread to cut or sample every frame

    // bind the function 'func_decode0' to cpu2
    CPU_ZERO(&mask);
    CPU_SET(2, &mask);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);
    pthread_create(&thread[2], &attr, func_decode0, NULL);    // create a thread to decode the QRcode of even frame

    // bind the function 'func_decode1' to cpu3
    CPU_ZERO(&mask);
    CPU_SET(3, &mask);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);
    pthread_create(&thread[3], &attr, func_decode1, NULL);    // create a thread to decode the QRcode of odd frame

    pthread_join(thread[0], NULL);
}
/*
 * retval: -1 set width fail
 *         -2 set height fail
 *         -3 set fourcc fail
 *          0 set camera success
 */
int set_camera(cv::VideoCapture cap)
{
    int ret = 0;
    // set width
    if (!cap.set(cv::CAP_PROP_FRAME_WIDTH, CAM_WIDTH)) {
        ret = -1;
        goto error_occur;
    }
    // set height
    if (!cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)) {
        ret = -2;
        goto error_occur;
    }
    // set fourcc
    if (!cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'))) {
        ret = -3;
        goto error_occur;
    }
    return ret;
error_occur:
    return ret;
}

/*
 * Function: QueryDevices
 * Description: Query the devices
 * Input: platform_id, out_dev_list
 * Output: None
*/
int32_t QueryDevices(cl_platform_id platform_id, std::vector<XClDeviceInfo> &out_dev_list)
{
    cl_int status = CL_SUCCESS;
    cl_uint num_devices = 0;
    cl_device_id* devices = NULL;

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Get the devices failed" << std::endl;
        return -1;
    }

    devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
    if (devices == NULL) {
        std::cerr << "Error: Allocate memory for devices failed" << std::endl;
        return -1;
    }

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Get the devices failed" << std::endl;
        return -1;
    }

    for (int i = 0; i < num_devices; i++) {
        XClDeviceInfo dev_info;
        dev_info.id = devices[i];
        status = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_info.device_type, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the device info failed" << std::endl;
            return -1;
        }

        status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &dev_info.vendor_id, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the device info failed" << std::endl;
            return -1;
        }

        status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dev_info.workitem_max_dim, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the device info failed" << std::endl;
            return -1;
        }

        status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &dev_info.max_workgroup_size, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the device info failed" << std::endl;
            return -1;
        }

        status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &dev_info.max_mem_size, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the device info failed" << std::endl;
            return -1;
        }
        out_dev_list.push_back(dev_info);
    }
        return 0;
}

/*
 * Function: QueryPlatforms
 * Description: Query the platforms
 * Input: None
 * Output: None
*/
int32_t QueryPlatforms()
{
    cl_int status = CL_SUCCESS;
    cl_uint num_platforms = 0;
    cl_platform_id* platforms = NULL;

    status = clGetPlatformIDs(0, NULL, &num_platforms); // get the number of platforms
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Get the platforms failed" << std::endl;
        return -1;
    }

    platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (platforms == NULL) {
        std::cerr << "Error: Allocate memory for platforms failed" << std::endl;
        return -1;
    }

    status = clGetPlatformIDs(num_platforms, platforms, NULL);  // get the ID of the platforms
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Get the platforms failed" << std::endl;
        return -1;
    }

    for (int i = 0; i < num_platforms; i++) {   // get the information of the platforms
        XClPlatformInfo platform_info;
        platform_info.id = platforms[i];
        char buffer[1024];
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(buffer), buffer, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the platform info failed" << std::endl;
            return -1;
        }
        platform_info.profile = buffer;

        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the platform info failed" << std::endl;
            return -1;
        }
        platform_info.version = buffer;

        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the platform info failed" << std::endl;
            return -1;
        }
        platform_info.name = buffer;

        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
        if (status != CL_SUCCESS) {
            std::cerr << "Error: Get the platform info failed" << std::endl;
            return -1;
        }
        platform_info.vendor = buffer;

        // get the devices of the platform
        QueryDevices(platforms[i], platform_info.devices);
        g_platforms.push_back(platform_info);
    }
    return 0;
}

int32_t LoadClFileToString(const string &cl_file_path, string &out_string)
{
    ifstream file(cl_file_path.c_str());
    if (!file.is_open())
    {
        cerr << "Error: Open the file failed" << endl;
        return -1;
    }

    stringstream ss;
    ss << file.rdbuf();
    out_string = ss.str();
    return 0;
}

/*
 * Function:
 * Description: calculate QRcode's rvecs & tvecs
    * Input: corners, board_size, camera_matrix, dist_coeffs
    * Output: rvecs, tvecs
*/
void calc_r_t(vector<vector<cv::Point2f>> corners, cv::Size board_size, cv::Mat camera_matrix, cv::Mat dist_coeffs, vector<cv::Mat>& rvecs, vector<cv::Mat>& tvecs)
{
    vector<cv::Point3f> obj_points(4);
    obj_points[0] = cv::Point3f(-board_size.width / 2.f, board_size.height / 2.f, 0);   // left up corner
    obj_points[1] = cv::Point3f(-board_size.width / 2.f, -board_size.height / 2.f, 0);  // left down corner
    obj_points[2] = cv::Point3f(board_size.width / 2.f, -board_size.height / 2.f, 0);   // right down corner
    obj_points[3] = cv::Point3f(board_size.width / 2.f, board_size.height / 2.f, 0);    // right up corner


    for (int i = 0; i < corners.size(); i++) {
        vector<cv::Point2f> corner = corners[i];
        cv::Mat rvec, tvec;
        cv::solvePnP(obj_points, corner, camera_matrix, dist_coeffs, rvec, tvec);
        rvecs.push_back(rvec);
        tvecs.push_back(tvec);
    }
}
