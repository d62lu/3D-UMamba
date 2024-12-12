"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.DALESBlockDataLoader_v3 import DALESDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from ErrorMatrix import ConfusionMatrix
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import time
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ground','vegetation','cars','trucks','powerlines','fences', 'poles','buildings']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_semseg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=150, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='2,3', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=8192, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=5, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_category', type=int, default=8, help='num_category')
    parser.add_argument('--weighted_loss', type=bool, default=False, help='weighted loss')


    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('dales_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # root = 'data/rs_data/'
    root = root = '../data/DALESObjects/input_0.100/DALES/Block_s20_min4096_norm_enhance/' #rs_40000_m
    
    NUM_CLASSES = 8
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    fps_n_list = [512, 128, 32]

    print("start loading training data ...")
    TRAIN_DATASET = DALESDataset(split='train', data_root=root, fps_n_list = fps_n_list, label_number = NUM_CLASSES, npoints = NUM_POINT)
    print("start loading test data ...")
    TEST_DATASET = DALESDataset(split='test',data_root=root, fps_n_list = fps_n_list, label_number = NUM_CLASSES, npoints = NUM_POINT)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=False,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=False)
    #weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    
    classifier = MODEL.get_model(NUM_CLASSES, fps_n_list).cuda()
    classifier = DataParallel(classifier)
    criterion = MODEL.get_loss().cuda()
    weight = torch.tensor(TRAIN_DATASET.labelweights).cuda()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if args.weighted_loss == True:
        print("Use weighted loss ...")
        criterion = MODEL.get_loss_weighted().cuda()
    classifier.apply(inplace_relu)
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/pre_model.pth')
        #start_epoch = checkpoint['epoch']
        start_epoch = 0
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)
    
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate/100)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    best_ave_F1_score = 0

    print('Start Tranining...')
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        
        '''Adjust learning rate and BN momentum'''
        
        #scheduler.step(best_instance_acc)
        logger.info('Learning rate is: %f' %(optimizer.state_dict()['param_groups'][0]['lr']))
        
        #lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        #log_string('Learning rate:%f' % lr)
        #for param_group in optimizer.param_groups:
            #param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target, fps_index_array, series_idx_arrays) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        #for i, (points, target) in enumerate(trainDataLoader):
            start_time1 = time.time()
            optimizer.zero_grad()
            
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            fps_index_array, series_idx_arrays = fps_index_array.long().cuda(), series_idx_arrays.long().cuda()
            points = points.transpose(2, 1)
            
            start_time = time.time()
            pre = classifier(points, fps_index_array, series_idx_arrays)
            end_time = time.time()
            # log_string('total training time: %f' % (end_time - start_time))

            pre = pre.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            
            loss = criterion(pre, target, weight)
            loss.backward()
            optimizer.step()

            pred_choice = pre.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
            end_time1 = time.time()
            # log_string('total all time: %f' % (end_time1 - start_time1))

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        scheduler.step()

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            #torch.save(state, savepath)
            #log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            #labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            inputs_list = []
            gt_list = []
            pred_list = []
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            catfile = os.path.join('/lustre04/scratch/dening/dening/DALES/names.txt')
            cat_names = [line.rstrip() for line in open(catfile)]
            confusion = ConfusionMatrix(num_classes=args.num_category, labels=cat_names)
            for i, (points, target, fps_index_array, series_idx_arrays) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                input_points = points
                inputs_list.append(input_points)
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                fps_index_array, series_idx_arrays = fps_index_array.long().cuda(), series_idx_arrays.long().cuda()
                points = points.transpose(2, 1)
                
                start_time = time.time()
                pre = classifier(points, fps_index_array, series_idx_arrays)
                end_time = time.time()
                #print('inference time:', end_time - start_time)
                # sys.stdout.flush()

                seg_pred = pre
                pred_val = seg_pred.contiguous().cpu().data.numpy()

                pre = pre.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy() #B,N
                batch_xyz_label = np.concatenate((input_points, np.expand_dims(batch_label, axis=-1)), axis=-1)
                gt_list.append(batch_xyz_label)
                target = target.view(-1, 1)[:, 0]
                loss = criterion(pre, target, weight)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2) #B,N
                pred_xyz_label = np.concatenate((input_points, np.expand_dims(pred_val, axis=-1)), axis=-1)
                pred_list.append(pred_xyz_label)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                # tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                # labelweights += tmp
                pred_val_flatten = pred_val.reshape(-1,1) # BN,1
                batch_label_flatten = batch_label.reshape(-1,1) # BN,1
                confusion.update(pred_val_flatten, batch_label_flatten)

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            # confusion.plot()
            ave_F1_score, miou, acc = confusion.summary()

            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point avg class IoU-2: %f' % (miou))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
   
            
            log_string('eval ave_F1_score: %f' % (ave_F1_score))
            # log_string('each category acc: %f' % (table))
            
            # iou_per_class_str = '------- IoU --------\n'
            # # for l in range(NUM_CLASSES):
            # #     iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
            # #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
            # #         total_correct_class[l] / float(total_iou_deno_class[l]))

            # log_string(iou_per_class_str)
            # log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            # log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
            #if ave_F1_score >= best_ave_F1_score:
                best_iou = mIoU
                #best_ave_F1_score = ave_F1_score
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ave_F1_score': ave_F1_score,
                }
                torch.save(state, savepath)
                log_string('Saving model....')

                # inputs = np.concatenate(inputs_list,0) # B N 3
                # gts = np.concatenate(gt_list,0) # B N 4
                # preds = np.concatenate(pred_list,0) # B N 4
                # for txt_i in range(inputs.shape[0]):
                #     input_ = inputs[txt_i,:,:]
                #     gt = gts[txt_i,:,:]
                #     pred = preds[txt_i,:,:]
                #
                #     input_name = 'dt_test_13/input/input_' + str(txt_i) +'.txt'
                #     gt_name = 'dt_test_13/gt/gt_' + str(txt_i) +'.txt'
                #     pred_name = 'dt_test_13/pred/pred_' + str(txt_i) +'.txt'
                #
                #     np.savetxt(input_name, input_)
                #     np.savetxt(gt_name, gt)
                #     np.savetxt(pred_name, pred)

            log_string('Best mIoU: %f' % best_iou)
            #log_string('Best_ave_F1_score: %f' % best_ave_F1_score)
        global_epoch += 1


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    main(args)
