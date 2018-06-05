#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <ctime>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include <sys/time.h>
#include <time.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "geometry_msgs/Twist.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

static const std::string RGB_WINDOW = "RGB Image window";
//static const std::string DEPTH_WINDOW = "DEPTH Image window";

#define Max_linear_speed 0.4
#define Min_linear_speed 0.2
#define Min_distance 1.2
#define Max_distance 5.0
#define Max_rotation_speed 0.8
#define ERROR_distance 1.0

#define k1 1
#define k2 9 
#define u_center 322.5//camera相对坐标u0 (pixel)
#define focus_X 576.7 //camera焦距 fx (pixel)

float linear_speed;
float target_linear_speed_x ;
float target_linear_speed_y ;
float target_vel;
float rotation_speed;
float control_speed;
float control_turn;

float target_pos_x ;
float target_pos_y ;

double time_update ;
double time_ros;
float cyt,cyt_last,dcyt ;
float dcxt;

float distance,distance_last;
//速度相对于目标距离的变化率
//float k_linear_speed = (Max_linear_speed - Min_linear_speed) / (Max_distance - Min_distance);
//
//float h_linear_speed = Min_linear_speed - k_linear_speed * Min_distance;

//float k_rotation_speed = 0.004;
//float h_rotation_speed_left = 1.2;
//float h_rotation_speed_right = 1.36;

int ERROR_OFFSET_X_left1 = 100;
int ERROR_OFFSET_X_left2 = 300;
int ERROR_OFFSET_X_right1 = 340;
int ERROR_OFFSET_X_right2 = 540;

int center_x;
int center_x_last;

cv::Mat rgbimage;
cv::Mat depthimage;
cv::Rect selectRect;
cv::Point origin;
cv::Point p0;
cv::Point px;
cv::Point py;
cv::Rect result;
cv::Rect result_last;//保留上一帧结果框

bool select_flag = false;
bool bRenewROI = false;  // the flag to enable the implementation of KCF algorithm for the new chosen ROI
bool bBeginKCF = false;
bool enable_get_depth = false;

bool HOG = true;                       //是否使用hog特征
bool FIXEDWINDOW = false;              //是否使用修正窗口
bool MULTISCALE = true;                //是否使用多尺度
bool SILENT = true;                    //是否不做显示
bool LAB = true;                      //是否使用LAB颜色

std::ofstream out_vel("/home/kkycj/turtlebot/turtlebot/turtlebot_kcf_ros/tracker_kcf_ros/velocity.txt",std::ios::app);
std::ofstream out_dist("/home/kkycj/turtlebot/turtlebot/turtlebot_kcf_ros/tracker_kcf_ros/distance.txt",std::ios::app);
std::ofstream out_linear("/home/kkycj/turtlebot/turtlebot/turtlebot_kcf_ros/tracker_kcf_ros/linear_speed.txt",std::ios::app);
std::ofstream out_rotation("/home/kkycj/turtlebot/turtlebot/turtlebot_kcf_ros/tracker_kcf_ros/rotation_speed.txt",std::ios::app);
std::ofstream out_dcxt("/home/kkycj/turtlebot/turtlebot/turtlebot_kcf_ros/tracker_kcf_ros/dcxt.txt",std::ios::app);
std::ofstream out_dcyt("/home/kkycj/turtlebot/turtlebot/turtlebot_kcf_ros/tracker_kcf_ros/dcyt.txt",std::ios::app);
std::string temp;

// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

float dist_val[5] ;

//float convert to string
std::string Convert(float Num)
{
	std::ostringstream oss;
	oss<<Num;
	std::string str(oss.str());
	return str;
}

std::string datetime()
{  
    time_t now = time(0);// 基于当前系统的当前日期/时间  
    tm *ltm = localtime(&now);  
  
    char iyear[50],imonth[50],iday[50],ihour[50],imin[50],isec[50];  
    sprintf(iyear, "%d",1900 + ltm->tm_year );  
    sprintf(imonth, "%02d", 1 + ltm->tm_mon );  
    sprintf(iday, "%02d", ltm->tm_mday );  
    sprintf(ihour, "%02d", ltm->tm_hour );  
    sprintf(imin, "%02d",  ltm->tm_min);  
    sprintf(isec, "%02d",  ltm->tm_sec);  
  
    std::vector<std::string> sDate{iyear, imonth, iday};  
    std::vector<std::string> sTime{ihour, imin, isec};  
    std::string myDate = boost::algorithm::join(sDate, "-") ;  
    std::string myTime = boost::algorithm::join(sTime, ":") ;  
    std::vector<std::string> sDateTime{myDate, myTime};  
    std::string myDateTime = boost::algorithm::join(sDateTime, " ") ;  
    return myDateTime;  
}  


void onMouse(int event, int x, int y, int, void*)
{
    if (select_flag)
    {
        selectRect.x = MIN(origin.x, x);        
        selectRect.y = MIN(origin.y, y);
        selectRect.width = abs(x - origin.x);   
        selectRect.height = abs(y - origin.y);
        //设定ROI区域，和窗口画面求交集 col列数 rows行数
        selectRect &= cv::Rect(0, 0, rgbimage.cols, rgbimage.rows);
    }
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        bBeginKCF = false;  
        select_flag = true; 
        origin = cv::Point(x, y);       
        //x,y左上角坐标;宽;高
        selectRect = cv::Rect(x, y, 0, 0);
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = false;
        bRenewROI = true;
    }
}

//define a timer
double get_wall_time()  
{  
    struct timeval time ;  
    if (gettimeofday(&time,NULL))
    {  
        return 0;  
    }  
    return (double)time.tv_sec + (double)time.tv_usec * .000001;  
}  

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber depth_sub_;
  
public:
  ros::Publisher pub;

  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_color", 1, 
      &ImageConverter::imageCb, this);
    // usb_cam_version
    //image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, 
    //  &ImageConverter::imageCb, this);
    depth_sub_ = it_.subscribe("/camera/depth/image", 1, 
      &ImageConverter::depthCb, this);
    pub = nh_.advertise<geometry_msgs::Twist>("tracker/cmd_vel", 1000);

    cv::namedWindow(RGB_WINDOW);
  //  cv::namedWindow(DEPTH_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(RGB_WINDOW);
    //cv::destroyWindow(DEPTH_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    // wiki.ros.org  cv_bridge 把ros的image信息转换成opencvImage
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }


    //把image的内容粘贴到rgbimage
    cv_ptr->image.copyTo(rgbimage);

    cv::setMouseCallback(RGB_WINDOW, onMouse, 0);

    if(bRenewROI)
    {
        // if (selectRect.width <= 0 || selectRect.height <= 0)
        // {
        //     bRenewROI = false;
        //     //continue;
        // }
        tracker.init(selectRect, rgbimage);
        out_rotation << datetime() << std::endl;
        out_linear << datetime() << std::endl;
        out_vel << datetime() << std::endl;
    	out_dist << datetime() << std::endl;
        out_dcxt << datetime() << std::endl;
        out_dcyt << datetime() << std::endl;
        
        result_last = selectRect;
        bBeginKCF = true;
        bRenewROI = false;
        enable_get_depth = false;
    }

    if(bBeginKCF)
    {
        //开始计时
    //    double start_time = get_wall_time();

        //结果框 update()基于当前帧更新目标位置
        result = tracker.update(rgbimage); 
        //结束计时
      //  double end_time = get_wall_time();

        //更新时间以秒为单位 计时器以毫秒为单位
        //time_update = ( end_time - start_time )/1000;

        

        //在rgbimage上绘制矩形框 紫色
        cv::rectangle(rgbimage, result, cv::Scalar( 0, 255, 255 ), 1, 8 );
        enable_get_depth = true;
    }
    else
        //蓝色roi框 宽度thickness 2 其他默认参数
        cv::rectangle(rgbimage, selectRect, cv::Scalar(255, 0, 0), 2, 8, 0);

        p0 = cv::Point(100,100);
        std::string text_time = Convert(time_update);
        std::string text_select = "Please select a target." ;
    if (select_flag)
        {
        cv::putText(rgbimage,text_select,p0,cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,255,255),2,8,0);	
        }
    cv::imshow(RGB_WINDOW, rgbimage);
    cv::waitKey(1);
  }

  void depthCb(const sensor_msgs::ImageConstPtr& msg)
  {
  	cv_bridge::CvImagePtr cv_ptr;
  	try
  	{
  		cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_32FC1);
  		cv_ptr->image.copyTo(depthimage);
  	}
  	catch (cv_bridge::Exception& e)
  	{
  		ROS_ERROR("Could not convert from '%s' to 'TYPE_32FC1'.", msg->encoding.c_str());
  	}

    if(enable_get_depth)
    {
      //at函数功能是访问矩阵元素 画面对称分布的5个点的距离信息
      dist_val[0] = depthimage.at<float>(result.y+result.height/3 , result.x+result.width/3) ;
      dist_val[1] = depthimage.at<float>(result.y+result.height/3 , result.x+2*result.width/3) ;
      dist_val[2] = depthimage.at<float>(result.y+2*result.height/3 , result.x+result.width/3) ;
      dist_val[3] = depthimage.at<float>(result.y+2*result.height/3 , result.x+2*result.width/3) ;
      dist_val[4] = depthimage.at<float>(result.y+result.height/2 , result.x+result.width/2) ;

      int num_depth_points = 5;
       
      for(int i = 0; i < 5; i++)
      {
        //取大于0.4 小于10.0的距离值
        if(dist_val[i] > 0.4 && dist_val[i] < 10.0)
          //叠加符合的距离值
          distance += dist_val[i];
        else
          //不符合则点数量减1
          num_depth_points--;
      }

      //取平均距离
      distance /= num_depth_points;

	  //calculate target linear speed
      
      center_x = result.x + result.width/2; //(pixel)
      center_x_last = result_last.x + result_last.width/2;
      float xd = Min_distance;

      //保留上一帧
      result_last = result;
      std::cout <<  "distance = " << distance << std::endl;
      out_dist << distance << std::endl;

      dcxt = fabs(distance_last - distance); // (m)
      cyt = - distance * (center_x - u_center ) / focus_X;
      cyt_last = - distance_last * (center_x_last - u_center ) / focus_X;
      dcyt = fabs(cyt_last - cyt);
      std::cout << "dcxt = " << dcxt << "  dcyt = " << dcyt << std::endl;
      out_dcxt << dcxt << std::endl;
      out_dcyt << dcyt << std::endl;

      std::cout << "distance_last = " << distance_last << std::endl;
	  distance_last = distance ;
 		
 		//std::cout << "time_update = " << time_update << std::endl;    
      target_linear_speed_y = dcyt / time_ros;
      target_linear_speed_x = dcxt / time_ros;
      
      linear_speed = target_linear_speed_x + k1 * ( distance - xd) + cyt * (k2 * cyt + target_linear_speed_y) / distance;
     
      std::cout << "target_linear_speed_x = " << target_linear_speed_x << "  target_linear_speed_y = " << target_linear_speed_y << std::endl;
      target_vel = sqrt(target_linear_speed_x * target_linear_speed_x + target_linear_speed_y * target_linear_speed_y);
      out_vel << target_vel << std::endl;

      //std::cout << "dcxt = " << dcxt << "  dcyt = " << dcyt << std::endl;
      //calculate rotation speed
      
      //if (center_x > ERROR_OFFSET_X_left2 && center_x < ERROR_OFFSET_X_right1)
      //  rotation_speed = 0;
      //else
        rotation_speed = (k2 * cyt + target_linear_speed_y) / distance;
      
      //超过最大距离以最大速度前进
      if(linear_speed > Max_linear_speed)
        linear_speed = Max_linear_speed;

      //如果旋转速度过大，则后退
      if (rotation_speed > Max_rotation_speed )
      {
       rotation_speed = Max_rotation_speed;
       // rotation_speed = 0;
      //  linear_speed = - Max_linear_speed;	
      }
      if (rotation_speed < -Max_rotation_speed )
      {
       rotation_speed = -Max_rotation_speed;
       // rotation_speed = 0;
      //  linear_speed = - Max_linear_speed;  
      }
      if (distance < Min_distance && distance > ERROR_distance)
      	linear_speed = 0;
	
	  if (distance < ERROR_distance)
	  	linear_speed = - Max_linear_speed ;

	  //缓慢减速beta
	  if (linear_speed > control_speed)
		control_speed = MIN( linear_speed, control_speed + 0.02 );	
      else if (linear_speed < control_speed)
        control_speed = MAX( linear_speed, control_speed - 0.02 );
      else
        control_speed = linear_speed;

      if (rotation_speed > control_turn)
        control_turn = MIN( rotation_speed, control_turn + 0.1 );
      else if (rotation_speed < control_turn)
        control_turn = MAX( rotation_speed, control_turn - 0.1 );
      else
        control_turn = rotation_speed;

    	  //停止运行
	  if (center_x < ERROR_OFFSET_X_left1 || center_x > ERROR_OFFSET_X_right2)
      {
		linear_speed = - Max_linear_speed;
     }
      if ( std::isinf(distance) != 0)
      {
      	bBeginKCF = false;
      	//select_flag = true;
      }
      else if (std::isnan(dcxt) ==1 || std::isnan(dcyt)==1 )
      {
       	bBeginKCF = false;
        //select_flag = true;
        control_turn = 0 ;
        control_speed = 0 ;
	  }
      

      px = cv::Point(0,20);
      py = cv::Point(0,40);

	  std::string text_target_x = Convert(target_linear_speed_x);
	  std::string text_target_y = Convert(target_linear_speed_y);
      //cv::putText(rgbimage,text_target_x,px,cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255),1,8,0);
      //cv::putText(rgbimage,text_target_y,py,cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255),1,8,0);      

      std::cout <<  "linear_speed = " << linear_speed << "  rotation_speed = " << rotation_speed << std::endl;
      out_linear << linear_speed << std::endl;
      out_rotation << rotation_speed << std::endl;
	  // std::cout <<  dist_val[0]  << " / " <<  dist_val[1] << " / " << dist_val[2] << " / " << dist_val[3] <<  " / " << dist_val[4] << std::endl;
      //std::cout <<  "distance = " << distance << std::endl;
    }

  	//cv::imshow(DEPTH_WINDOW, depthimage);
  	cv::waitKey(1);
  }
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "kcf_tracker");
	ImageConverter ic;
  
	while(ros::ok())
	{
		double start_ros = get_wall_time();

		ros::spinOnce();

    	geometry_msgs::Twist twist;
    	twist.linear.x = control_speed; 
    	twist.linear.y = 0; 
    	twist.linear.z = 0;
    	twist.angular.x = 0; 
    	twist.angular.y = 0; 
    	twist.angular.z = control_turn;
    	ic.pub.publish(twist);
    //ROS_INFO("target_linear_speed_x = %f , target_linear_speed_y = %f",target_linear_speed_x,target_linear_speed_y);
    //ROS_INFO("dcxt = %f, dcyt = %f",dcxt,dcyt);
		
		double end_ros = get_wall_time();
		time_ros = end_ros - start_ros;

		ROS_INFO("time_ros = %lf",time_ros);

		if (cvWaitKey(33) == 'q')
    	{
    		out_vel.close();
    		out_dist.close();
    		out_linear.close();
  	  		out_rotation.close();
    		break;
    	}
	}

	return 0;
}
