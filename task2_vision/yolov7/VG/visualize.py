# 0 0.51 0.6466666666666666 0.1025 0.41333333333333333 
# 0 0.333125 0.6391666666666667 0.07125 0.43166666666666664 
import os


if __name__ == '__main__':
    train_img_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/train/images'
    train_label_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/train/labels'
    
    train_img = os.listdir(train_img_path)
    train_label = os.listdir(train_img_path)
    
    val_img_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/valid/images'
    val_label_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/valid/labels'
    
    val_img = os.listdir(val_img_path)
    val_label = os.listdir(val_img_path)
    import pdb; pdb.set_trace()



