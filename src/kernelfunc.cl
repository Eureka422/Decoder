
__kernel void image_filter(
	const __global uchar*   in_img_data,
	const int               pxl_bytes,
	const int               img_line_bytes, 
	__global int*		  	border,
	const float		  		block_size,
	__global uchar*         sample_img_data  
  )
{
const int width = 800; 
const int height = 600; 
const int center_x = 2 * get_global_id(0);   // 这里的值就是获取第一个维度当前数据项索引
const int center_y = 2 * get_global_id(1);   // 这里的值就是获取第二个维度当前数据项索引
*(sample_img_data + (center_y/2 * img_line_bytes/2 + center_x/2 * pxl_bytes)) = (uchar)*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes));

int start = 4 * block_size;
int offset_0 = (int)block_size;
int k = 6;
if (*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)>=40 && center_x >=start&&center_y >=start&&center_x<width-start&&center_y<height-start)
{
	//int offset_45 = (int)(0.707 * block_size);
 	//int score = 0;
	/*if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y-offset_45)*img_line_bytes + (center_x-offset_45)*pxl_bytes)) score++;
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y)*img_line_bytes + (center_x-offset_0)*pxl_bytes)) score++;
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y+offset_45)*img_line_bytes + (center_x-offset_45)*pxl_bytes)) score++;
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y+offset_0)*img_line_bytes + center_x*pxl_bytes)) score++;
	
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y+offset_45)*img_line_bytes + (center_x+offset_45)*pxl_bytes)) score++;
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y)*img_line_bytes + (center_x+offset_0)*pxl_bytes)) score++;
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y-offset_45)*img_line_bytes + (center_x+offset_45)*pxl_bytes)) score++;
	if(*(in_img_data + (center_y * img_line_bytes + center_x * pxl_bytes))*4<*(in_img_data + (center_y-offset_0)*img_line_bytes + center_x*pxl_bytes)) score++;*/
	//if(score>=6)
	//{
	int score2 = 0;
	
	//int offset2_0 = (int)(2 * block_size);
  	//int offset2_45 = (int)(1.414 * block_size);
  	int offset2_0 = (int)(1.75 * block_size);
  	int offset2_45 = (int)(1.237 * block_size);
  	
	if(*(in_img_data + (center_y-offset2_45)*img_line_bytes + (center_x-offset2_45)*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;	// 左上角小于中心
	if(*(in_img_data + center_y*img_line_bytes + (center_x-offset2_0)*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;					// 正下方左侧小于中心
	if(*(in_img_data + (center_y+offset2_45)*img_line_bytes + (center_x-offset2_45)*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;	// 左下角小于中心
	if(*(in_img_data + (center_y+offset2_0)*img_line_bytes + center_x*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;

	if(*(in_img_data + (center_y+offset2_45)*img_line_bytes + (center_x+offset2_45)*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;
	if(*(in_img_data + center_y*img_line_bytes + (center_x+offset2_0)*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;
	if(*(in_img_data + (center_y-offset2_45)*img_line_bytes + (center_x+offset2_45)*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;
	if(*(in_img_data + (center_y-offset2_0)*img_line_bytes + center_x*pxl_bytes)*k<*(in_img_data + center_y * img_line_bytes + center_x * pxl_bytes)) score2++;

	if(score2>=6)
	{
		int score3 = 0;
		
		//int offset3_0 = (int)(4 * block_size);
  		//int offset3_45 = (int)(2.828 * block_size);
  		int offset3_0 = (int)(3.5 * block_size);
  		int offset3_45 = (int)(2.475 * block_size);
  		
  		if(*(in_img_data + (center_y-offset2_45)*img_line_bytes + (center_x-offset2_45)*pxl_bytes)*k<*(in_img_data + (center_y-offset3_45)*img_line_bytes + (center_x-offset3_45)*pxl_bytes)) score3++;
		if(*(in_img_data + center_y*img_line_bytes + (center_x-offset2_0)*pxl_bytes)*k<*(in_img_data + center_y*img_line_bytes + (center_x-offset3_0)*pxl_bytes)) score3++;
		if(*(in_img_data + (center_y+offset2_45)*img_line_bytes + (center_x-offset2_45)*pxl_bytes)*k<*(in_img_data + (center_y+offset3_45)*img_line_bytes + (center_x-offset3_45)*pxl_bytes)) score3++;
		if(*(in_img_data + (center_y+offset2_0)*img_line_bytes + center_x*pxl_bytes)*k<*(in_img_data + (center_y+offset3_0)*img_line_bytes + center_x*pxl_bytes)) score3++;
	
		if(*(in_img_data + (center_y+offset2_45)*img_line_bytes + (center_x+offset2_45)*pxl_bytes)*k<*(in_img_data + (center_y+offset3_45)*img_line_bytes + (center_x+offset3_45)*pxl_bytes)) score3++;
		if(*(in_img_data + center_y*img_line_bytes + (center_x+offset2_0)*pxl_bytes)*k<*(in_img_data + center_y*img_line_bytes + (center_x+offset3_0)*pxl_bytes)) score3++;
		if(*(in_img_data + (center_y-offset2_45)*img_line_bytes + (center_x+offset2_45)*pxl_bytes)*k<*(in_img_data + (center_y-offset3_45)*img_line_bytes + (center_x+offset3_45)*pxl_bytes)) score3++;
		if(*(in_img_data + (center_y-offset2_0)*img_line_bytes + center_x*pxl_bytes)*k<*(in_img_data + (center_y-offset3_0)*img_line_bytes + center_x*pxl_bytes)) score3++;
		
		if(score3>=6)
		{			
			atomic_min(border, 	center_x + 1.5*block_size); 
			atomic_max(border + 1, center_x - 1.5*block_size);
			atomic_min(border + 2, center_y + 1.5*block_size);
			atomic_max(border + 3, center_y - 1.5*block_size);
		}
	}
	//}
}
}

