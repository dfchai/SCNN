from __future__ import print_function
import os, time, cv2, sys, math, csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import gdal

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train" or "test" mode. ')
parser.add_argument('--dataset', type=str, default="L8", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=2048, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=2048, help='Width of cropped input image to network')
parser.add_argument('--step_height', type=int, default=2048, help='Height of cropped input image to network')
parser.add_argument('--step_width', type=int, default=2048, help='Width of cropped input image to network')
parser.add_argument('--num_samples', type=int, default=100, help='Width of cropped input image to network')
parser.add_argument('--dropout', type=float, default=0.5, help='Width of cropped input image to network')
parser.add_argument('--learningrate', type=float, default=0.0001, help='Width of cropped input image to network')
parser.add_argument('--decay', type=float, default=0.95, help='decay')
parser.add_argument('--netindex', type=int, default=13, help='layers and depths')


args = parser.parse_args()

def LOG(X, f=None):
	time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	if not f:
		print(time_stamp + " " + X)
	else:
		f.write(time_stamp + " " + X)

def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

def compareImage(gt,label,class_labels_list):
    num_classes=len(class_labels_list)
    count=np.zeros((num_classes+1, num_classes+1))
    for i in range(0, num_classes):
        li=class_labels_list[i]
        lbi=cv2.compare(label,li,cv2.CMP_EQ)
        for j in range(0, num_classes):
            lj=class_labels_list[j]
            gtj=cv2.compare(gt,lj,cv2.CMP_EQ)
            cij=cv2.bitwise_and(lbi,gtj)
            count[i][j]=cv2.countNonZero(cij)
            count[i][num_classes]=count[i][num_classes]+count[i][j]
            count[num_classes][j]=count[num_classes][j]+count[i][j]
    for j in range(0, num_classes):
        count[num_classes][num_classes]=count[num_classes][num_classes]+count[num_classes][j]
    return count

def outputcount(target,count,class_names_list,id):
    num_classes=len(class_names_list)

    Recall=np.zeros(num_classes)
    Precision=np.zeros(num_classes)
    Fscore=np.zeros(num_classes)

    target.write("%s,"%id)
    for j in range(0, num_classes):
        target.write("%s," % class_names_list[j])
    target.write("Accuracy\n")
    for i in range(0, num_classes):
        target.write("%s," % class_names_list[i])
        for j in range(0, num_classes):
            target.write("%d," % count[i][j])
        if count[i][num_classes]==0:
            Precision[i]==0
        else:
            Precision[i]=1.0*count[i][i]/count[i][num_classes]
        target.write("%f\n" %(Precision[i]))
    target.write("Accuracy,")
    corr=0
    for j in range(0, num_classes):
        corr=corr+count[j][j]
        if count[num_classes][j]==0:
            Recall[j]=0
        else:
            Recall[j]=1.0*count[j][j]/count[num_classes][j]
        target.write("%f," %(Recall[j]))
    target.write("%f\n" %(1.0*corr/count[num_classes][num_classes]))
    target.write("Fscore,")
    for j in range(0, num_classes):
        den=Precision[j]+Recall[j]
        if den < 0.00000001:
            Fscore[j]=0
        else:
            Fscore[j]=2.0*(Precision[j]*Recall[j])/den
        target.write("%f," %(Fscore[j]))
    target.write(",\n")
    return Recall, Precision, Fscore

def Drawcurve(num_epochs, avg_scores_per_epoch, avg_loss_per_epoch, path):
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    num_epochs=len(avg_loss_per_epoch)
    
    ax1.plot(range(num_epochs), avg_scores_per_epoch, linewidth=3)
    ax1.set_title("Average validation accuracy vs epochs", fontsize=24)
    ax1.set_xlabel("Epoch", fontsize=24)
    ax1.set_ylabel("Avg. val. accuracy", fontsize=24)

    plt.savefig("%s/%s"%(path, 'accuracy_vs_epochs.png'))

    plt.clf()

    ax1 = fig.add_subplot(111)
    
    ax1.plot(range(num_epochs), avg_loss_per_epoch, linewidth=3)
    ax1.set_title("Average loss vs epochs", fontsize=24)
    ax1.set_xlabel("Epoch", fontsize=24)
    ax1.set_ylabel("Current loss", fontsize=24)

    plt.savefig("%s/%s"%(path, 'loss_vs_epochs.png'))

    return

def build_net_smooth(img, dropout_p=0.5, scope=None):
    net = img
    net = slim.conv2d(net, 64, [1,1], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(net)
    if dropout_p != 0.0:
        net = slim.dropout(net, keep_prob=(1.0-dropout_p))

    net = slim.conv2d(net, 2, [1, 1], padding='VALID', activation_fn=None, normalizer_fn=None)
    net = slim.conv2d(net, 2, [3, 3], padding='VALID', activation_fn=None, normalizer_fn=None)
    label = tf.argmax(net, 3)

    return net, label
    
def load_image_scene_l8(covername,scenename):
    mv=[]
    j=0
    for i in list(range(1,8)) + list(range(9, 12)):
        if i>9:
            imgfile=("../cloud/L8/L8T/%s/%s/%s_bt_band%d.tif" %(covername,scenename,scenename, i))
        else:
            imgfile=("../cloud/L8/L8T/%s/%s/%s_toa_band%d.tif" %(covername,scenename,scenename, i))
        srcRaster = gdal.Open(imgfile)
        img = srcRaster.ReadAsArray()
        img = img + 32768
        mv.append(img)
        j=j+1

    image=cv2.merge(mv)

    imgfile=("../cloud/L8/L8T/%s/%s/%s_BQA.TIF"% (covername,scenename,scenename))
    bqaRaster = gdal.Open(imgfile)
    label_bqa = bqaRaster.ReadAsArray()
    labelRaster = srcRaster
    transform_label = bqaRaster.GetGeoTransform()
    
    imgfile=("../cloud/L8/L8T/%s/%s/%s.TIF"% (covername,scenename,scenename))
    gtRaster = gdal.Open(imgfile)
    label_gt = gtRaster.ReadAsArray()
    transform_gt = gtRaster.GetGeoTransform()
    height=label_gt.shape[0]
    width=label_gt.shape[1]

    x0 = int((transform_label[0]-transform_gt[0])/transform_label[1])
    y0 = int((transform_label[3]-transform_gt[3])/transform_label[5])
    x1 = x0+labelRaster.RasterXSize
    y1 = y0+labelRaster.RasterYSize

    c0=0
    c1=x1-x0
    r0=0
    r1=y1-y0
    if x0<0:
        c0=-x0
        x0=0
    if y0<0:
        r0=-y0
        y0=0
    if x1>width:
        c1=c1+width-x1
        x1=width
    if y1>height:
        r1=r1+height-y1
        y1=height

    return image, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1
    
def load_image_scene_l7(covername,scenename):
    mv=[]
    for i in list(range(1,6)):
        imgfile=("../cloud/L7/L7T/%s/%s/%s_toa_band%d.tif" %(covername,scenename,scenename, i))
        srcRaster = gdal.Open(imgfile)
        img = srcRaster.ReadAsArray()
        mv.append(img)

    imgfile=("../cloud/L7/L7T/%s/%s/%s_bt_band%d.tif" %(covername,scenename,scenename, 6))
    srcRaster = gdal.Open(imgfile)
    img = srcRaster.ReadAsArray()
    mv.append(img)

    imgfile=("../cloud/L7/L7T/%s/%s/%s_toa_band%d.tif" %(covername,scenename,scenename, 7))
    srcRaster = gdal.Open(imgfile)
    img = srcRaster.ReadAsArray()
    mv.append(img)

    imgfile=("../cloud/L7/L7T/%s/%s/%s_BQA.TIF"% (covername,scenename,scenename))
    bqaRaster = gdal.Open(imgfile)
    label_bqa = bqaRaster.ReadAsArray()
    labelRaster = srcRaster
    transform_label = bqaRaster.GetGeoTransform()
    
    maskpath=("../cloud/L7/L7T/%s/%s"% (covername,scenename))
    for maskpath_sub in os.listdir(maskpath):
        if maskpath_sub.find("newmask2015")>=0:
            maskpath_ful=os.path.join(maskpath, maskpath_sub)
    gtRaster = gdal.Open(maskpath_ful)
    label_gt = gtRaster.ReadAsArray()
    transform_gt = gtRaster.GetGeoTransform()
    height=label_gt.shape[0]
    width=label_gt.shape[1]

    image=cv2.merge(mv)

    x0 = int((transform_label[0]-transform_gt[0])/transform_label[1])
    y0 = int((transform_label[3]-transform_gt[3])/transform_label[5])
    x1 = x0+labelRaster.RasterXSize
    y1 = y0+labelRaster.RasterYSize

    c0=0
    c1=x1-x0
    r0=0
    r1=y1-y0
    if x0<0:
        c0=-x0
        x0=0
    if y0<0:
        r0=-y0
        y0=0
    if x1>width:
        c1=c1+width-x1
        x1=width
    if y1>height:
        r1=r1+height-y1
        y1=height

    return image, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1
    
def load_image_scene_gf1(subpath):
    imgfile=("GF1/GF1/%s.tiff" %(subpath[0]))
    imgRaster = gdal.Open(imgfile)
    img = imgRaster.ReadAsArray()
    img=np.swapaxes(img, 0, 1)
    img=np.swapaxes(img, 1, 2)

    maskfile=("GF1/GF1/%s_ReferenceMask.tif" %(subpath[0]))
    maskRaster = gdal.Open(maskfile)
    mask = maskRaster.ReadAsArray()
    return img, mask
    
def SaveImages(gt_image, pred_image, filename):
    comptmp=cv2.compare(gt_image,pred_image,cv2.CMP_NE)
    cv2.imwrite("%s_gt.png"%(filename),np.uint8(gt_image))
    cv2.imwrite("%s_pred.png"%(filename),np.uint8(pred_image))
    cv2.imwrite("%s_comp.png"%(filename),np.uint8(comptmp))
    return

def Test_Scene_l7(testpath, testset):
    cwd = os.getcwd()
    group_path=os.path.join(cwd + "/L7/L7_Samples/groups.csv")

    target=open("%s/%s_scores.csv"%(testpath,testset),'w')
    #covertarget
    covercount=np.zeros((num_classes+1, num_classes+1))
    totalcount=np.zeros((num_classes+1, num_classes+1))
    totalimage=0
    totaltime=0
    with open(group_path, newline='') as groupfile:
        file_reader = csv.reader(groupfile, delimiter=',')
        covername=""
        for row in file_reader:
            if row[9]==testset:
                path_c = testpath + "/" + row[0]
                if not os.path.isdir(path_c):
                    os.makedirs(path_c)

                path_s = path_c  + "/" +  row[8]
                if not os.path.isdir(path_s):
                    os.makedirs(path_s)

                scenetarget=open("%s/%s_scores_scene.csv"%(path_s,testset),'w')
                scenecount=np.zeros((num_classes+1, num_classes+1))

                sys.stdout.write("\r\rRunning test image %s %s           "%(row[0], row[8]))
                sys.stdout.flush()

                image_orig, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1 = load_image_scene_l7(row[0],row[8])
                st = time.time()

                h = image_orig.shape[0]
                w = image_orig.shape[1]
                pred_label=np.zeros([h,w],np.uint8)
                image=np.zeros([h,w,7],np.float32)

                yrang=list(range(crop_height_2, h-crop_height-crop_height_2, crop_height))
                yrang.append(h-crop_height-crop_height_2)
                xrang=list(range(crop_width_2, w-crop_width-crop_width_2, crop_width))
                xrang.append(w-crop_width-crop_width_2)

                for j in range(7):
                    image[:,:,j] = (image_orig[:,:,j]-mean[j])/sdev[j]
                for y in yrang:
                    for x in xrang:
                        img=image[y-crop_height_2:y+crop_height+crop_height_2, x-crop_width_2:x+crop_width+crop_width_2]
                        
                        image_batch=np.expand_dims(img, axis=0)    
                        output_label = sess.run(predlabel,feed_dict={input:image_batch})
                        output_label = np.uint8((output_label[0,:,:]+1)*50)
                        
                        pred_label[y:y+crop_height, x:x+crop_width]=output_label
                
                label_pred=pred_label

                label_gt = (label_gt==64)*50+(label_gt==128)*50+(label_gt==191)*100+(label_gt==192)*100+(label_gt==255)*100
                label_gt = np.uint8(label_gt)

                label_bqa=label_bqa[r0:r1,c0:c1]
                label_pred=label_pred[r0:r1,c0:c1]
                label_gt=label_gt[y0:y1,x0:x1]
                label_gt=label_gt*(label_bqa!=1)
                label_pred=label_pred*(label_gt>0)

                totaltime =totaltime + time.time() -st
                totalimage=totalimage + 1

                scenecount = compareImage(label_gt,label_pred,class_labels_list)
                outputcount(scenetarget,scenecount,class_names_list,row[8])

                scenetarget.close()

                if covername=="":
                    covername=row[0]
                    covertarget=open("%s/%s_scores_%s.csv"%(path_c,testset,covername),'w')
                elif covername!=row[0]:
                    outputcount(target, covercount, class_names_list, covername)
                    totalcount=totalcount+covercount
                    covercount=np.zeros((num_classes+1, num_classes+1))

                    covertarget.close()
                    covername=row[0]
                    covertarget=open("%s/%s_scores_%s.csv"%(path_c,testset,covername),'w')

                outputcount(covertarget, scenecount, class_names_list, row[8])
                covercount=covercount+scenecount

    outputcount(target, covercount, class_names_list, covername)
    totalcount=totalcount+covercount

    outputcount(target,totalcount,class_names_list,"Total")
    target.write("%s,%s,%s\n" % ("totaltime","totalimage","totaltime/totalimage"))
    target.write("%f,%f,%f\n" % (totaltime,totalimage,totaltime/totalimage))

    covertarget.close()
    target.close()
    return totalcount

def Test_Scene_l8(testpath, testset):
    cwd = os.getcwd()
    group_path=os.path.join(cwd + "/L8/L8_Samples/groups.csv")

    target=open("%s/%s_scores.csv"%(testpath,testset),'w')
    #covertarget
    covercount=np.zeros((num_classes+1, num_classes+1))
    totalcount=np.zeros((num_classes+1, num_classes+1))
    totalimage=0
    totaltime=0
    with open(group_path, newline='') as groupfile:
        file_reader = csv.reader(groupfile, delimiter=',')
        covername=""
        for row in file_reader:
            if row[11]==testset:
                path_c = testpath + "/" + row[0]
                if not os.path.isdir(path_c):
                    os.makedirs(path_c)

                path_s = path_c  + "/" +  row[10]
                if not os.path.isdir(path_s):
                    os.makedirs(path_s)

                scenetarget=open("%s/%s_scores_scene.csv"%(path_s,testset),'w')
                scenecount=np.zeros((num_classes+1, num_classes+1))

                sys.stdout.write("\r\rRunning test image %s %s           "%(row[0], row[10]))
                sys.stdout.flush()

                image_orig, label_gt, label_bqa, r0, r1, c0, c1, y0, y1, x0, x1 = load_image_scene_l8(row[0],row[10])

                st = time.time()

                h = image_orig.shape[0]
                w = image_orig.shape[1]
                pred_label=np.zeros([h,w],np.uint8)
                image=np.zeros([h,w,10],np.float32)

                yrang=list(range(crop_height_2, h-crop_height-crop_height_2, crop_height))
                yrang.append(h-crop_height-crop_height_2)
                xrang=list(range(crop_width_2, w-crop_width-crop_width_2, crop_width))
                xrang.append(w-crop_width-crop_width_2)

                for j in range(10):
                    image[:,:,j] = (image_orig[:,:,j]-mean[j])/sdev[j]

                for y in yrang:
                    for x in xrang:
                        img=image[y-crop_height_2:y+crop_height+crop_height_2, x-crop_width_2:x+crop_width+crop_width_2]
                        
                        image_batch=np.expand_dims(img, axis=0)    
                        output_label = sess.run(predlabel,feed_dict={input:image_batch})
                        output_label = np.uint8((output_label[0,:,:]+1)*50)
                        
                        pred_label[y:y+crop_height, x:x+crop_width]=output_label
                
                label_pred=pred_label

                label_gt = (label_gt==50)*50+(label_gt==100)*50+(label_gt==150)*100+(label_gt==200)*100
                label_gt = np.uint8(label_gt)

                label_bqa=label_bqa[r0:r1,c0:c1]
                label_pred=label_pred[r0:r1,c0:c1]
                label_gt=label_gt[y0:y1,x0:x1]
                label_gt=label_gt*(label_bqa!=1)
                label_pred=label_pred*(label_gt>0)

                totaltime =totaltime + time.time() -st
                totalimage=totalimage + 1

                scenecount = compareImage(label_gt,label_pred,class_labels_list)

                outputcount(scenetarget,scenecount,class_names_list,row[10])

                scenetarget.close()

                if covername=="":
                    covername=row[0]
                    covertarget=open("%s/%s_scores_%s.csv"%(path_c,testset,covername),'w')
                elif covername!=row[0]:
                    outputcount(target, covercount, class_names_list, covername)
                    totalcount=totalcount+covercount
                    covercount=np.zeros((num_classes+1, num_classes+1))

                    covertarget.close()
                    covername=row[0]
                    covertarget=open("%s/%s_scores_%s.csv"%(path_c,testset,covername),'w')

                outputcount(covertarget, scenecount, class_names_list, row[1])
                covercount=covercount+scenecount
                

    outputcount(target, covercount, class_names_list, covername)
    totalcount=totalcount+covercount

    outputcount(target,totalcount,class_names_list,"Total")
    target.write("%s,%s,%s\n" % ("totaltime","totalimage","totaltime/totalimage"))
    target.write("%f,%f,%f\n" % (totaltime,totalimage,totaltime/totalimage))

    covertarget.close()
    target.close()
    return totalcount

def Test_Scene_gf1(testpath, testset):
    cwd = os.getcwd()
    group_path=os.path.join(cwd + "/GF1/GF1_Samples/groups.csv")

    target=open("%s/%s_scores.csv"%(testpath,testset),'w')
    totalcount=np.zeros((num_classes+1, num_classes+1))
    totalimage=0
    totaltime=0
    with open(group_path, newline='') as groupfile:
        file_reader = csv.reader(groupfile, delimiter=',')
        for row in file_reader:
            if row[1]==testset:
                path_s = testpath  + "/" +  row[0]
                if not os.path.isdir(path_s):
                    os.makedirs(path_s)

                scenetarget=open("%s/%s_scores_scene.csv"%(path_s,testset),'w')
                scenecount=np.zeros((num_classes+1, num_classes+1))

                sys.stdout.write("\r\rRunning test image %s           "%(row[0]))
                sys.stdout.flush()

                image_orig, label_gt = load_image_scene_gf1(row)
                label_bqa = label_gt
                h = label_bqa.shape[0]
                w = label_bqa.shape[1]
                st = time.time()

                h = image_orig.shape[0]
                w = image_orig.shape[1]
                pred_label=np.zeros([h,w],np.uint8)
                image=np.zeros([h,w,4],np.float32)

                yrang=list(range(crop_height_2, h-crop_height-crop_height_2, crop_height))
                yrang.append(h-crop_height-crop_height_2)
                xrang=list(range(crop_width_2, w-crop_width-crop_width_2, crop_width))
                xrang.append(w-crop_width-crop_width_2)

                j=0
                for i in range(4):
                    image[:,:,j] = (image_orig[:,:,j]-mean[j])/sdev[j]
                    j=j+1
                for y in yrang:
                    for x in xrang:
                        img=image[y-crop_height_2:y+crop_height+crop_height_2, x-crop_width_2:x+crop_width+crop_width_2]
                        
                        image_batch=np.expand_dims(img, axis=0)    
                        output_label = sess.run(predlabel,feed_dict={input:image_batch})
                        output_label = np.uint8((output_label[0,:,:]+1)*50)
                        
                        pred_label[y:y+crop_height, x:x+crop_width]=output_label
                
                label_pred=pred_label

                label_gt = (label_gt==1)*50+(label_gt==128)*50+(label_gt==255)*100
                label_gt = np.uint8(label_gt)

                label_pred=label_pred*(label_gt>0)

                totaltime =totaltime + time.time() -st
                totalimage=totalimage + 1

                scenecount = compareImage(label_gt,label_pred,class_labels_list)
                outputcount(scenetarget,scenecount,class_names_list,row[0])
                outputcount(target, scenecount, class_names_list, row[0])
                
                scenetarget.close()

                totalcount=totalcount+scenecount

    outputcount(target,totalcount,class_names_list,"Total")
    target.write("%s,%s,%s\n" % ("totaltime","totalimage","totaltime/totalimage"))
    target.write("%f,%f,%f\n" % (totaltime,totalimage,totaltime/totalimage))
    target.close()
    return totalcount

exppath=args.dataset+"/Experiment"
if not os.path.isdir(exppath):
    os.makedirs(exppath)
checkpointpath=exppath+"/checkpoints"
if not os.path.isdir(checkpointpath):
    os.makedirs(checkpointpath)

num_classes = 2
class_names_list=list()
class_names_list.append('clear')
class_names_list.append('cloud')
class_labels_list=list()
class_labels_list.append(50)
class_labels_list.append(100)

crop_height=args.crop_height
crop_width=args.crop_width
step_height=args.step_height
step_width=args.step_width

crop_height_2=1
crop_width_2=1

mean=np.loadtxt("%s/%s_Samples/mean.txt"%(args.dataset,args.dataset))
sdev=np.loadtxt("%s/%s_Samples/sdev.txt"%(args.dataset,args.dataset))

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#cpu=tf.config.list_physical_devices("CPU")
#tf.config.set_visible_devices(cpu)
sess=tf.Session(config=config)
if args.dataset=="L7":
    numchannel=7
elif args.dataset=="L8":
    numchannel=10
elif args.dataset=="GF1":
    numchannel=4

print("Preparing the model ...")

input = tf.placeholder(tf.float32,shape=[None, None, None, numchannel])
gt = tf.placeholder(tf.uint8,shape=[None, None, None])

network = None
init_fn = None

indices = gt
output=tf.one_hot(indices, num_classes)
indices0=tf.to_float(indices)
indices1=tf.expand_dims(indices0, axis=3)

dropout_p=args.dropout
learning_rate=args.learningrate

network, predlabel = build_net_smooth(input, dropout_p, scope=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))

varall=[var for var in tf.trainable_variables()]
opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.995).minimize(loss, var_list=varall)


saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

count_params()

if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = checkpointpath + "/latest_model.ckpt"

if args.mode == "train":

    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Num Epochs -->", args.num_epochs)
    print("")

    mv=[]
    if args.dataset=="L8":

        for j in range(10):
            imgfile=("%s/%s_Samples/band_%d.tif"%(args.dataset,args.dataset,j))
            srcRaster = gdal.Open(imgfile)
            img = srcRaster.ReadAsArray()
            img = (img-mean[j])/sdev[j]
            mv.append(img)
        image=cv2.merge(mv)
        image=image[:,0:args.num_samples*11,0:10]
        image_batch=np.expand_dims(image, axis=0)

        labfile=("%s/%s_Samples/label.tif"%(args.dataset,args.dataset))
        labRaster = gdal.Open(labfile)
        labe = labRaster.ReadAsArray()
        labe = (labe==50)*0+(labe==100)*0+(labe==150)*1+(labe==200)*1

    elif args.dataset=="L7":
        for j in range(7):
            imgfile=("%s/%s_Samples/band_%d.tif"%(args.dataset,args.dataset,j))
            srcRaster = gdal.Open(imgfile)
            img = srcRaster.ReadAsArray()
            img = (img-mean[j])/sdev[j]
            mv.append(img)
        image=cv2.merge(mv)
        image=image[:,0:args.num_samples*11,:]
        image_batch=np.expand_dims(image, axis=0)

        labfile=("%s/%s_Samples/label.tif"%(args.dataset,args.dataset))
        labRaster = gdal.Open(labfile)
        labe = labRaster.ReadAsArray()
        labe = (labe==64)*0+(labe==128)*0+(labe==191)*1+(labe==192)*1+(labe==255)*1

    elif args.dataset=="GF1":
        for j in range(4):
            imgfile=("%s/%s_Samples/band_%d.tif"%(args.dataset,args.dataset,j))
            srcRaster = gdal.Open(imgfile)
            img = srcRaster.ReadAsArray()
            img = (img-mean[j])/sdev[j]
            mv.append(img)
        image=cv2.merge(mv)
        image=image[:,0:args.num_samples*11,:]
        image_batch=np.expand_dims(image, axis=0)

        labfile=("%s/%s_Samples/label.tif"%(args.dataset,args.dataset))
        labRaster = gdal.Open(labfile)
        labe = labRaster.ReadAsArray()
        labe = (labe==1)*0+(labe==128)*0+(labe==255)*1

    h=image.shape[0]
    w=image.shape[1]
    yrang=range(0, h, 11)
    xrang=range(0, w, 11)

    imgs=[]
    labs=[]
    for y in yrang:
        for x in xrang:
            img=image[y+5-crop_height_2:y+5+crop_height_2+1, x+5-crop_width_2:x+5+crop_width_2+1]
            lab=labe[y+5:y+6, x+5:x+6]
            imgs.append(img)
            labs.append(lab)
    image_batch = np.stack(imgs, axis=0)
    label_batch = np.stack(labs, axis=0)

    avg_loss_per_epoch = []
    avg_scores_per_epoch = []
    avg_scoresnb_per_epoch = []

    # Do the training here
    for epoch in range(0, args.num_epochs):
        current_losses = []

        for i in range(200):
            st = time.time()
            _,current=sess.run([opt,loss],feed_dict={input:image_batch, gt: label_batch})

            current_losses.append(current)

        string_print = "Epoch = %d Current = %.2f Time = %.2f"%(epoch,current,time.time()-st)
        LOG(string_print)

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)
        
        # Create directories if needed
        val_path="%s/%04d"%(checkpointpath,epoch)
        if not os.path.isdir(val_path):
            os.makedirs(val_path)

        saver.save(sess,model_checkpoint_name)
        #if epoch >= args.num_epochs-5:
        #    saver.save(sess, val_path + "/latest_model.ckpt")

        #if you have prepared the dataset, you can uncomment the following to do validation at the end of each epoch
        '''
        if args.dataset=="L7":
            totalcount = Test_Scene_l7(val_path,"Val")
        elif args.dataset=="L8":
            totalcount = Test_Scene_l8(val_path,"Val")
        elif args.dataset=="GF1":
            totalcount = Test_Scene_gf1(val_path,"Val")

        corr=0
        for j in range(0, num_classes):
            corr=corr+totalcount[j][j]
        '''
        avg_score=1.0#*corr/totalcount[num_classes][num_classes]
        avg_scores_per_epoch.append(avg_score)

        print("avg_score %f, mean loss %f" %(avg_score, mean_loss))

        Drawcurve(args.num_epochs,avg_scores_per_epoch,avg_loss_per_epoch,checkpointpath)
        
elif args.mode == "test":
    print("\n***** Begin testing *****")
    print("Dataset -->", args.dataset)
    print("")

    print('Loaded latest model at checkpoint')
    saver.restore(sess, model_checkpoint_name)

    test_path=exppath+"/Test"
    if not os.path.isdir("%s"%(test_path)):
        os.makedirs("%s"%(test_path))
    if args.dataset=="L7":
        Test_Scene_l7(test_path,"Test")
    elif args.dataset=="L8":
        Test_Scene_l8(test_path,"Test")
    elif args.dataset=="GF1":
        Test_Scene_gf1(test_path,"Test")

else:
    ValueError("Invalid mode selected.")