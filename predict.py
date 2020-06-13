from __future__ import division
import argparse
from utils.util import *
from utils.dataset import *
import cv2
from PIL import Image
import torch
# from torchvision import transforms
# def resize(image, size):
#     image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
#     return image


# def yolo_prediction(model, device, image,class_names):
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     imgs = transforms.ToTensor()(Image.fromarray(image))
#     c, h, w = imgs.shape
#     img_sacle = [w / 416, h / 416, w / 416, h / 416]
#     imgs = resize(imgs, 416)
#     imgs = imgs.unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(imgs)
#         outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.45)
#
#     # print(outputs)
#     objects = []
#     try:
#         outputs = outputs[0].cpu().data
#         for i, output in enumerate(outputs):
#             item = []
#             item.append(class_names[int(output[-1])])
#             print(class_names[int(output[-1])])
#             item.append(float(output[4]))
#             print(float(output[4]))
#             box = [int(value * img_sacle[i]) for i, value in enumerate(output[:4])]
#             x1,y1,x2,y2 = box
#             x = int((x2+x1)/2)
#             y = int((y1+y2)/2)
#             w = x2-x1
#             h = y2-y1
#             item.append([x,y,w,h])
#             print([x,y,w,h])
#             objects.append(item)
#             print(item)
#     except:
#         pass
#     return objects

def detect(source,half=False,weights='weights/yolov5m.pt',imgsz=int(640),devices='0,1'):
    # imgsz = \ opt.img_size
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    # imgs = transforms.ToTensor()(Image.fromarray(image))
    # c, h, w = imgs.shape
    # # img_sacle = [w / 416, h / 416, w / 416, h / 416]
    # imgs = resize(imgs, 416)
    # print('okkkk')
    # imgs = imgs.unsqueeze(0).to(devices)
    # Initialize
    device = torch_utils.select_device(devices)
    # # if os.path.exists(out):
    # #     shutil.rmtree(out)  # delete output folder
    # # os.makedirs(out)  # make new output folder
    #
    # # Load model
    # # google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    # # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    #
    #
    model.fuse()
    model.to(device).eval()

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    #     modelc.to(device).eval()
    #
    # # Half precision
    # half = half and device.type != 'cpu'  # half precision only supported on CUDA
    # if half:
    #     model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    #     save_img = True
    dataset = Load_Images(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    # t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for img, im0s in dataset:

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]
        # t2 = torch_utils.time_synchronized()

        # # to float
        # if half:
        #     pred = pred.float()
        #
        # # Apply NMS
        pred = non_max_suppression(pred, conf_thres=float(0.4), iou_thres=float(0.5),
                                   fast=True, classes='', agnostic=False)
        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)
        objects=[]
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            # s, im0 = '', im0s
            # # save_path = str(Path(out) / Path(p).name)
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                    # print('names[int(c)]:', names[int(c)])    #object names
                    # print('det:',det)
                    # s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # print('s:', s)

    #             # Write results
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    x = int((x2 + x1) / 2)
                    y = int((y1 + y2) / 2)
                    w = x2 - x1
                    h = y2 - y1

                    item=[]
                    label = '%s %.2f' % (names[int(cls)], conf)
                    # print('label', label)
                    objectname = '%s' % (names[int(cls)])
                    objectpro = '%.2f' % (conf)
                    item.append(objectname)
                    item.append(float(objectpro))
                    item.append([x,y,w,h])
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    # print('xyxy', int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    objects.append(item)
                #     print('object:',objects)
                # print('objectsss:', objects)
                return objects
    #                 if save_txt:  # Write to file
    #                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
    #                         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
    #
                    # if save_img or view_img:  # Add bbox to image

    #
    #         # Print time (inference + NMS)
    #         # print('%sDone. (%.3fs)' % (s, t2 - t1))
    #
    #         # Stream results
    #         if view_img:
    #             cv2.imshow(p, im0)
    #             if cv2.waitKey(1) == ord('q'):  # q to quit
    #                 raise StopIteration
    #
    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'images':
    #                 cv2.imwrite(save_path, im0)
    #             else:
    #                 if vid_path != save_path:  # new video
    #                     vid_path = save_path
    #                     if isinstance(vid_writer, cv2.VideoWriter):
    #                         vid_writer.release()  # release previous video writer
    #
    #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
    #                 vid_writer.write(im0)
    #
    # if save_txt or save_img:
    #     # print('Results saved to %s' % os.getcwd() + os.sep + out)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    # parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # opt = parser.parse_args()
    # print(opt)

    # vid=cv2.VideoCapture('inference/images/video-02.mp4')
    # while vid.isOpened():
    #     ret ,fram =vid.read()
    #     cv2.imshow('1',fram)
    #     cv2.waitKey(1000)
    #     detect(source=fram)
    # with torch.no_grad():
    #     detect()


