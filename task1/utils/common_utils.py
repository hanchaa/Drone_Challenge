import cv2

class MatchImageSizeTo(object):
    def __init__(self, size=1080):
        self.size=size

    def __call__(self, img):
        H, W = img.shape

        if H>=W:
            W_size = int(W/H * self.size * (1920/1450))
            # W_size = int(W/H * self.size)
            img_new = cv2.resize(img, (W_size, self.size))
        else:
            H_size = int(H/W * self.size * (1450/1920))
            # H_size = int(H/W * self.size)
            img_new = cv2.resize(img, (self.size, H_size))
        
        return img_new

def save_tag_match_imgs(image, detections, path):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    tag_family = detections.tag_family
    tag_id = detections.tag_id
    center = detections.center
    corners = detections.corners

    center = (int(center[0]), int(center[1]))
    corner_01 = (int(corners[0][0]), int(corners[0][1]))
    corner_02 = (int(corners[1][0]), int(corners[1][1]))
    corner_03 = (int(corners[2][0]), int(corners[2][1]))
    corner_04 = (int(corners[3][0]), int(corners[3][1]))

    cv2.circle(image, (center[0], center[1]), 5, (0, 180, 255), 1)
    cv2.line(image, (corner_01[0], corner_01[1]), (corner_02[0], corner_02[1]), (0, 255, 0), 2)
    cv2.line(image, (corner_02[0], corner_02[1]), (corner_03[0], corner_03[1]), (0, 255, 0), 2)
    cv2.line(image, (corner_03[0], corner_03[1]), (corner_04[0], corner_04[1]), (0, 255, 0), 2)
    cv2.line(image, (corner_04[0], corner_04[1]), (corner_01[0], corner_01[1]), (0, 255, 0), 2)

    cv2.putText(image, str(tag_family)+'_'+str(tag_id), (corner_01[0] - 5, corner_01[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite((path+'/test.png'), image)

def save_match_imgs(frame_idx, imgs, path):  # for dubugging
    for i in range(0, len(imgs)):
        cv2.imwrite((path+'/frame_'+'{}'+'.png').format(i+frame_idx[0]), imgs[i])