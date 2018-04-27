#../../caffe-segnet/build/tools/caffe train -gpu 4 -weights ../../Pretrained/classfication_finetune.caffemodel -solver faceinpaint_segnet_wo_lastbn_solver.prototxt 2>&1 | tee ../log/treelog_faceinpaint_regression_finetune_scale_gray_wolastbn_1e-6.log
# -weights ./slx_face/face.caffemodel -weights ./slx_face/clean.caffemode


#../../caffe/build/tools/caffe train -gpu 2 -solver Nobn_segnet_classification_adam_solver.prototxt -weights ../vgg_face_caffe/VGG_FACE.caffemodel,../slx_face/clean.caffemodel,../slx_face/face.caffemodel 2>&1 | tee ../log/Adam_Nobn_classification_scaleinput_vggface_finetune_1e-5.log

#../../caffe/build/tools/caffe train -gpu 2 -solver Tanh_segnet_classification_adam_solver.prototxt -weights ../vgg_face_caffe/VGG_FACE.caffemodel,../slx_face/clean.caffemodel,../slx_face/face.caffemodel 2>&1 | tee ../log/Adam_Tanh_classification_1e-4_multifeature_soft_withVGG_weight_1e-3.log

#../../caffe/build/tools/caffe train -gpu 0 -solver AAAI_6_classification_adam_solver.prototxt -snapshot ../model/AAAI_6_conv1_iter_20000.caffemodel 2>&1 | tee ../log/AAAI_6_conv1_finetune_2w.log


#Tanh_adam_classfication_1e-4_multifeature_soft_withVGG_weight_iter_40000.caffemodel

#for finetuning from a single loss model
../../caffe/build/tools/caffe train -gpu 6 -solver AAAI_ST_classification_adam_solver.prototxt -weights ../slx_face/8.5_fc_clean.caffemodel,../slx_face/8.5_fc_face.caffemodel 2>&1 | tee ../log/AAAI_ST_weighted_pixel_conv2_fc_Bloss_8_32.log
#../../caffe/build/tools/caffe train -gpu 6 -solver AAAI_ST_classification_adam_solver.prototxt -snapshot ../model/AAAI_ST_weighted_pixel_conv2_fc_Bloss_30_16_iter_90000.caffemodel 2>&1 | tee ../log/AAAI_ST_weighted_pixel_conv2_fc_Bloss_30_16_from9w.log

