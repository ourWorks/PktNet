# nohup python /my_data/project/python/classification/end/trainFusionModelByName.py && python /my_data/project/python/mail.py "融合模型" "训练完成" &
# tensorboard --logdir runs
# 如果不断拼接oneTestEpochOutputs，那么使用命令行调用本脚本时可能会出现RuntimeError: Pin memory thread exited unexpectedly有关

import csv
import gc
from datasetWithLoader.adni_pureSMRI import ADNIlist
import torch
import torch.nn as nn
import os
import pandas as pd
import torch.nn.functional as F
from models.factory import ModelFactory
# 计算指标
from evaluation.use_torchmetrics import EvaClassify
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
import argparse
from configs.several_log import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 检查是否出现nan
# torch.autograd.set_detect_anomaly(True)
from lightning.pytorch import Trainer, seed_everything
seed_everything(42, workers=True)


class Recorder:

    def __init__(self, writer, logger, BASE_DATASET_NAME, subTask, fiveFoldName, startEpochIndex, max_epochs, BEST_PTH_PATH, \
                    NUM_CLASSES, MODEL_NAME, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, TRAIN_NUM_WORKERS, TEST_NUM_WORKERS):

        super(Recorder, self).__init__()

        self.writer = writer
        self.logger = logger

        # 控制训练过程的参数
        self.BASE_DATASET_NAME = BASE_DATASET_NAME
        self.subTask = subTask
        self.fiveFoldName = fiveFoldName
        self.startEpochIndex = startEpochIndex
        self.max_epochs = max_epochs
        self.BEST_PTH_PATH = BEST_PTH_PATH
        self.NUM_CLASSES = NUM_CLASSES

        self.MODEL_NAME = MODEL_NAME
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.TEST_BATCH_SIZE = TEST_BATCH_SIZE
        self.TRAIN_NUM_WORKERS = TRAIN_NUM_WORKERS
        self.TEST_NUM_WORKERS = TEST_NUM_WORKERS
        
        # 控制多少次训练后验证一次
        self.test_interval = 1

        # 控制记录的参数
        self.best_accuracy = -1
        self.best_accuracy_epoch = -1
        
        self.SEN_inBestAccuracy = -1
        self.SPE_inBestAccuracy = -1
        self.AUC_inBestAccuracy = -1
        self.F1Score_inBestAccuracy = -1

        self.epoch_loss_values = []
        self.accuracy_values = []

        self.train_record = EvaClassify(NUM_CLASSES)
        self.test_record = EvaClassify(NUM_CLASSES)

    def recordInCsv(self, recordLine):

        # 设置CSV文件路径
        csv_file_folder = f'/my_data/project/python/classification/end/{self.BASE_DATASET_NAME}_results'
        os.makedirs(csv_file_folder, exist_ok=True)
        csv_file_path = f"{csv_file_folder}/result.csv"

        # 打开CSV文件，准备写入数据
        # 如果文件不存在，写入表头
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='a', newline='') as csvfile:
                # 创建CSV写入对象
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Task', 'Five Fold Name', 'Model_Name', 'ACC','SEN','SPE','AUC','F1-score', \
                'End Train Batch Num', 'Params Num'])
        
        with open(csv_file_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 将数据写入CSV文件
            csv_writer.writerow(recordLine)


# def eval(test_dataloader, device, MODEL_NAME, TEST_BATCH_SIZE, oneTestEpochOutputs, recorder.logger, trainEpochIndex, recorder.test_record, all_testLabels, accuracy_values, BEST_PTH_PATH, recorder.writer, fiveFoldName):

# @profile
def trainAndEval(model, optimizer, loss_function, object, recorder):

    if recorder.NUM_CLASSES==2:
        TERMINAL_RATE = 0.999
    else:
        TERMINAL_RATE = 0.997

    train_dataloader = object.getDataLoader(run_stage="train",batch_size=recorder.TRAIN_BATCH_SIZE, num_workers=recorder.TRAIN_NUM_WORKERS)
    test_dataloader = object.getDataLoader(run_stage="test",batch_size=recorder.TEST_BATCH_SIZE, num_workers=recorder.TEST_NUM_WORKERS)

    accuracy_values = []

    trainDF = pd.DataFrame(train_dataloader.dataset.data)
    all_trainLabels = torch.tensor(trainDF['label'].to_list()).to(device)

    testDF = pd.DataFrame(test_dataloader.dataset.data)
    all_testLabels = torch.tensor(testDF['label'].to_list()).to(device)

    OneEpochTrainPreds = torch.zeros((train_dataloader.dataset.__len__(), recorder.NUM_CLASSES), requires_grad=False).to(device)
    oneTestEpochOutputs = torch.zeros((test_dataloader.dataset.__len__(), recorder.NUM_CLASSES), requires_grad=False).to(device)

    # 如果最后5次准确率返回的数值都不发生变化，那就认为函数已经收敛，终止batch循环：使用负数切片，长度不够也=不会报错，还有记住切片操作的左边闭、右边开
    stable_train_acc_times = 0

    # 6、开始训练和测试
    # for trainEpochIndex in tqdm(range(startEpochIndex, max_epochs+1), desc="Training Epochs", leave=False):
    for trainEpochIndex in range(recorder.startEpochIndex, recorder.max_epochs+1):

        recorder.logger.info("-" * 10)
        recorder.logger.info(f"epoch {trainEpochIndex}/{recorder.max_epochs}")
        model.train()

        epoch_loss = 0
        step = 0

        repeat_epoch_time = 0

        # for train_idx,train_batch in enumerate(tqdm(train_dataloader, desc="one training batch", leave=True), start=0):
        for train_idx,train_batch in enumerate(train_dataloader, start=0):

            # 需要打印的步骤
            step += 1

            epoch_len = train_dataloader.dataset.__len__() // train_dataloader.batch_size # 【这里需要取上界】
            if train_dataloader.dataset.__len__() % train_dataloader.batch_size != 0:  # 如果余数不为0，表示需要向上取整
                epoch_len += 1

            train_inputs, train_labels = train_batch['sMRI'].to(device), train_batch['label'].to(device)
            # temp, train_labels = train_batch['sMRI'], train_batch['label'].to(device)
            # chooseIndex = torch.tensor([34, 35, 36, 37, 38, 39, 40, 41])
            # train_inputs = torch.index_select(temp, 1, chooseIndex).to(device)

            # gpu_tracker.track()

            # if(trainEpochIndex > 30):
            #     model.frozenEncoder()
            outputs = model(train_inputs, train_batch['Age'].to(device),train_batch['MMSE'].to(device))
            # model.upDateFeature(features, train_labels, train_idx)
            # nt_loss = nt_xent_loss(outputs, train_labels, 0.5)
            # nt_loss = compare_loss(features, train_labels)
            # loss = nt_loss + loss_function(outputs, train_labels.long()) # 返回的loss是一个metatensor
            # loss = nt_loss
            loss = loss_function(outputs, train_labels.long()) # 返回的loss是一个metatensor
            loss.backward()                       # 更新梯度矩阵：根据损失和学习率来更新
            optimizer.step()                      # 更新模型参数：利用计算的梯度来优化模型  
        
            # if(train_idx == len(train_dataloader)-1):
            # #     # # 用来测试梯度是否存在，顺便判断梯度是否回传
            #     for name, parms in model.named_parameters():
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        
            optimizer.zero_grad()  #zero_grad()只要不出现在loss.backward() 和 optimizer.step() 中间
            epoch_loss += loss.item()             # loss.item()才是需要的数值，epoch_loss是累加之前遍历epoch的损失值

            startIndex = train_idx*recorder.TRAIN_BATCH_SIZE
            currentBatchNum = train_inputs.shape[0]
            OneEpochTrainPreds[startIndex:startIndex+currentBatchNum, :] = outputs # 留意前面切片的写法
 
            # 打印每一次迭代时的损失：【训练损失通常是小批次，就一个batch_size大小的数据的损失】
            ##########################################可以看情况，统计几次的数据平均值######################################
            # recorder.logger.info(f"[{step}/{epoch_len}][{trainEpochIndex}/{recorder.max_epochs}], train_loss: {loss.item():.6f}, nt_loss:{nt_loss.item():.6f}")
            recorder.logger.info(f"[{step}/{epoch_len}][{trainEpochIndex}/{recorder.max_epochs}], train_loss: {loss.item():.6f}")
            recorder.writer.add_scalar("iter_loss", loss.item(), epoch_len * (trainEpochIndex-1) + step)

        recorder.train_record.evaluate(OneEpochTrainPreds, all_trainLabels)

        epoch_loss /= (step+repeat_epoch_time)    # 计算所有图像中得到的平均loss
        recorder.epoch_loss_values.append(epoch_loss)      # epoch_loss_values储存的一个item是batch_size张图像的平均loss
        
        # last_epoch_loss = epoch_loss             # 保存上一个epoch的loss
        recorder.logger.info(f"Epoch {trainEpochIndex} average train loss: {epoch_loss:.6f}")
        recorder.writer.add_scalar("train_loss", epoch_loss, trainEpochIndex )

        ### 【千万记住，下面的代码不要使用loss，不要参与训练过程，下面的代码是用来评估模型效果的 】
        if trainEpochIndex % recorder.test_interval == 0:

            model.eval()

            # for test_idx,test_batch in enumerate(tqdm(test_dataloader, desc="    test batch", leave=True), start=0):
            for test_idx,test_batch in enumerate(test_dataloader, start=0):

                ##### 和 bothMRIs 的区别4
                test_inputs = test_batch['sMRI'].to(device)

                with torch.no_grad():

                    oneTestBatchOutputs = model(test_inputs, test_batch['Age'].to(device),test_batch['MMSE'].to(device))

                    startIndex = test_idx*recorder.TEST_BATCH_SIZE
                    currentBatchNum = test_inputs.shape[0]  # 这个是本次迭代的样本数量，为了下面label的赋值而获取，不采用BATCH_SIZE是考虑最后一次迭代的batch_num可能不一样
                    oneTestEpochOutputs[startIndex:startIndex+currentBatchNum, :] = oneTestBatchOutputs # 留意前面切片的写法
                    
            recorder.logger.info(f"Current epoch: {trainEpochIndex}")
            
            # 计算指标,下面这两步是原子操作
            recorder.test_record.evaluate(oneTestEpochOutputs, all_testLabels)

            accuracy_values.append(recorder.test_record.ACC)

            if recorder.test_record.ACC > recorder.best_accuracy:

                recorder.best_accuracy = recorder.test_record.ACC
                recorder.best_accuracy_epoch = trainEpochIndex

                # 更新其它数值
                SEN_inBestAccuracy = recorder.test_record.SEN
                SPE_inBestAccuracy = recorder.test_record.SPE
                AUC_inBestAccuracy = recorder.test_record.AUC
                F1Score_inBestAccuracy = recorder.test_record.F1_Score
                
                # torch.save(model.state_dict(), BEST_PTH_PATH)
                torch.save({
                            'epoch': recorder.best_accuracy_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                            }, recorder.BEST_PTH_PATH)
                
                recorder.logger.info("saved new best accuracy model")

            recorder.logger.info(f"Current end epoch: {trainEpochIndex} current accuracy: {recorder.test_record.ACC:.6f} ")
            recorder.logger.info(f"Best accuracy: {recorder.best_accuracy:.6f} at epoch {recorder.best_accuracy_epoch}")

            recorder.writer.add_scalar("ACC", recorder.test_record.ACC, trainEpochIndex)
            recorder.writer.add_scalar("SEN", recorder.test_record.SEN, trainEpochIndex)
            recorder.writer.add_scalar("SPE", recorder.test_record.SPE, trainEpochIndex)
            recorder.writer.add_scalar("AUC", recorder.test_record.AUC, trainEpochIndex)
            recorder.writer.add_scalar("F1-score", recorder.test_record.F1_Score, trainEpochIndex)

            print(f"Eopch:{trainEpochIndex} train acc:{recorder.train_record.ACC}, test acc:{recorder.test_record.ACC}")
            recorder.logger.info(f"Eopch:{trainEpochIndex} train acc:{recorder.train_record.ACC}, test acc:{recorder.test_record.ACC}")

        if recorder.train_record.ACC > TERMINAL_RATE: # 需要满足连续5次的训练精度已判断已经收敛；否则就训练够200轮
            stable_train_acc_times += 1
        else:
            stable_train_acc_times = 0
        if stable_train_acc_times == 5:
            break

        gc.collect()
        torch.cuda.empty_cache()

    recorder.logger.info(f"Cross-testation {recorder.fiveFoldName} end:")
    recorder.logger.info(f"Training completed, recorder.best_accuracy: {recorder.best_accuracy:.6f} at epoch: {recorder.best_accuracy_epoch}")
    recorder.logger.info(f"######################################################################################\n")
    recorder.writer.close()
    recordLine = [recorder.subTask, recorder.fiveFoldName, recorder.MODEL_NAME, recorder.best_accuracy, SEN_inBestAccuracy, SPE_inBestAccuracy, AUC_inBestAccuracy, F1Score_inBestAccuracy, \
                len(accuracy_values), count_parameters(model=model) ]
    recorder.recordInCsv(recordLine)

def count_parameters(model):
    """统计模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description='My test (and eval) a model')
    parser.add_argument('--max_epochs', help='num of train batches', default=200)
    parser.add_argument('--model_name', help='Model Name', default="PktNet")
    parser.add_argument('--json_path', help='json file for dataset', 
                        default=f"/my_data/dataset/ADNI_1207/5_link_dataset/pureSMRI_1207_threeStage/multiClass_5Fold/CNvsMCIvsAD/fiveFold_3.json")
    parser.add_argument('--out_path', help='output path for LOG, PTH and recorder.writer', default=None)
    parser.add_argument('--num_classes', help='num of classes', default=3)
    parser.add_argument('--train_batch_size', help='batch_size for training', default = 3)
    parser.add_argument('--test_batch_size', help='batch_size for testing',default = 1)
    parser.add_argument('--pth', help='predTrained for model',default = None)
    parser.add_argument('--train_num_workers', help='the num of training workers',default = 0)
    parser.add_argument('--test_num_workers', help='the num of testing workers',default = 0)
    args = parser.parse_args()
    return args

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

def main():

    args = parse_args()

    JSON_PATH = args.json_path

    BASE_DATASET_NAME = JSON_PATH.split("/")[5]

    print("Hello World!")

    # 1、重要的输入参数写在这里
    NUM_CLASSES = int(args.num_classes)
    MODEL_NAME = args.model_name
    TRAIN_BATCH_SIZE = int(args.train_batch_size)
    TEST_BATCH_SIZE = int(args.test_batch_size)
    TRAIN_NUM_WORKERS = int(args.train_num_workers)
    TEST_NUM_WORKERS = int(args.test_num_workers)

    # 2、设置开始和结束的训练次数，以及加载模型、选择性加载模型权重
    startEpochIndex = 1
    max_epochs = int(args.max_epochs)
    # max_epochs = 2
    LOAD_PTH = args.pth

    # 3、加载模型、设置优化器、损失函数，并选择性恢复模型的权重信息、实例化实验记录对象
    model = ModelFactory().create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES)

    # model.apply(init_weights)

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    if LOAD_PTH is not None:
        checkpoint = torch.load(LOAD_PTH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 需要将optimizer里的数据加载到gpu
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        startEpochIndex = checkpoint['epoch'] + 1
        # loss = checkpoint['loss']
        # 记得删除，以便减少显存占用
        del checkpoint
        # recorder.logger.info(f"Continue training from epoch {startEpochIndex}...:")
        
    model.to(device)

    # 4、输出的文件夹：包含(1)可视化记录tensorboard recorder.writer、(2)权重文件的保存位置、(3)记录信息的训练日志log 以及初始化recoder用来管理recorder.writer、logger
    import datetime
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") # 获取当前日期和时间

    subTask = JSON_PATH.split('/')[-3] + '/' + JSON_PATH.split('/')[-2]
    subTask = subTask.replace('_5Fold','')
    fiveFoldName = os.path.basename(JSON_PATH).replace(".json","")
    OUT_PATH = f"{BASE_DATASET_NAME}_results/{subTask}"

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"{OUT_PATH}/runs/{MODEL_NAME}_{fiveFoldName}_{current_datetime}", comment=f"{MODEL_NAME}_batch-{TRAIN_BATCH_SIZE}_{optimizer.__class__.__name__}-{learning_rate}_{current_datetime}")
    
    os.makedirs(f"{OUT_PATH}/pths", exist_ok=True)
    BEST_PTH_PATH = f"{OUT_PATH}/pths/{MODEL_NAME}_{fiveFoldName}_{current_datetime}_bestTestAccuracy.pth"

    log_path = f'{OUT_PATH}/logs/{MODEL_NAME}_{fiveFoldName}_{current_datetime}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = get_logger(log_path)
    print(f"@training.log的位置在{log_path}")

    recorder = Recorder(writer, logger, BASE_DATASET_NAME, subTask, fiveFoldName, startEpochIndex, max_epochs, BEST_PTH_PATH, \
                    NUM_CLASSES, MODEL_NAME, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, TRAIN_NUM_WORKERS, TEST_NUM_WORKERS)


    # 5、创建数据加载器对象
    object = ADNIlist(jsonPath=JSON_PATH)

    recorder.logger.info(f"数据集：{BASE_DATASET_NAME}")
    recorder.logger.info(f"模型名称：{MODEL_NAME}")
    recorder.logger.info(f"使用优化器：{optimizer.__class__.__name__}")
    recorder.logger.info(f"初始学习率：{learning_rate}")
    recorder.logger.info(f"train_batch_size：{TRAIN_BATCH_SIZE}")
    recorder.logger.info(f"test_batch_size：{TEST_BATCH_SIZE}")
    recorder.logger.info(f"train_num_workers：{TRAIN_NUM_WORKERS}")
    recorder.logger.info(f"test_num_workers:{TEST_NUM_WORKERS}")
    recorder.logger.info(f"{fiveFoldName} start:")

    # try:
    trainAndEval(model, optimizer, loss_function, object, recorder)
    # except Exception as e:# 写一个except
    #     print(f"发生异常，信息是: {e}")

if __name__ == '__main__':
    main()