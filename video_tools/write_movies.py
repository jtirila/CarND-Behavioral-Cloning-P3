import cv2


def write_movie(imgs):
    height, width, layers = imgs[0].shape

    video = cv2.VideoWriter('/tmp/video.avi', -1, 1, (width, height), isColor=True)

    for img in imgs:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    return video
