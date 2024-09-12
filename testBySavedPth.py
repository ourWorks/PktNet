# nohup python /my_data/project/python/classification/end/trainFusionModelByName.py && python /my_data/project/python/mail.py "融合模型" "训练完成" &
# tensorboard --logdir runs
# 如果不断拼接oneTestEpochOutputs，那么使用命令行调用本脚本时可能会出现RuntimeError: Pin memory thread exited unexpectedly有关

import csv
import gc
# from preprocess.region_pureSMRI import ADNIlist
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

def recordInCsv(recordLine, BASE_DATASET_NAME):

    # 设置CSV文件路径
    csv_file_folder = f'testResults/{BASE_DATASET_NAME}_results'
    os.makedirs(csv_file_folder, exist_ok=True)
    csv_file_path = f"{csv_file_folder}/oneTimeResult.csv"

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


# def eval(test_dataloader, device, MODEL_NAME, TEST_BATCH_SIZE, oneTestEpochOutputs, recorder.logger, trainEpochIndex, test_record, all_testLabels, accuracy_values, BEST_PTH_PATH, recorder.writer, fiveFoldName):

# @profile
# def trainAndEval(model, optimizer, loss_function, object, recorder):
def testAndRecord(model, object, NUM_CLASSES, TEST_BATCH_SIZE, TEST_NUM_WORKERS, subTask, fiveFoldName, MODEL_NAME, BASE_DATASET_NAME):

    test_record = EvaClassify(NUM_CLASSES)

    test_dataloader = object.getDataLoader(run_stage="test",batch_size=TEST_BATCH_SIZE, num_workers=TEST_NUM_WORKERS)

    accuracy_values = []


    testDF = pd.DataFrame(test_dataloader.dataset.data)
    all_testLabels = torch.tensor(testDF['label'].to_list()).to(device)

    oneTestEpochOutputs = torch.zeros((test_dataloader.dataset.__len__(), NUM_CLASSES), requires_grad=False).to(device)

    model.eval()

    # for test_idx,test_batch in enumerate(tqdm(test_dataloader, desc="    test batch", leave=True), start=0):
    for test_idx,test_batch in enumerate(test_dataloader, start=0):

        ##### 和 bothMRIs 的区别4
        test_inputs = test_batch['sMRI'].to(device)

        with torch.no_grad():

            oneTestBatchOutputs = model(test_inputs, test_batch['Age'].to(device),test_batch['MMSE'].to(device))

            startIndex = test_idx*TEST_BATCH_SIZE
            currentBatchNum = test_inputs.shape[0]  # 这个是本次迭代的样本数量，为了下面label的赋值而获取，不采用BATCH_SIZE是考虑最后一次迭代的batch_num可能不一样
            oneTestEpochOutputs[startIndex:startIndex+currentBatchNum, :] = oneTestBatchOutputs # 留意前面切片的写法
    
    # 计算指标,下面这两步是原子操作
    test_record.evaluate(oneTestEpochOutputs, all_testLabels)
    accuracy_values.append(test_record.ACC)

    print(f"test acc:{test_record.ACC}")

    recordLine = [subTask, fiveFoldName, MODEL_NAME, test_record.ACC, test_record.SEN, test_record.SPE,test_record.AUC, test_record.F1_Score, count_parameters(model=model) ]
    recordInCsv(recordLine, BASE_DATASET_NAME)

def count_parameters(model):
    """统计模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description='My test (and eval) a model')
    parser.add_argument('--max_epochs', help='num of train batches', default=200)
    parser.add_argument('--model_name', help='Model Name', default="PktNet")
    parser.add_argument('--json_path', help='json file for dataset', 
                        default=f"data/pureSMRI_1207_threeStage/twoClass_5Fold/CNvsAD/fiveFold_3.json")
    parser.add_argument('--out_path', help='output path for LOG, PTH and recorder.writer', default=None)
    parser.add_argument('--num_classes', help='num of classes', default=2)
    parser.add_argument('--train_batch_size', help='batch_size for training', default = 3)
    parser.add_argument('--test_batch_size', help='batch_size for testing',default = 1)
    parser.add_argument('--pth', help='predTrained for model',default = f"/my_data/project/python/article/PktNet_result/ablation_pureSMRI_1207_threeStage_results/twoClass/CNvsAD/pths/PktNet_fiveFold_3_2024-08-31@20-01_bestTestAccuracy.pth")
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

    # 5、创建数据加载器对象
    object = ADNIlist(jsonPath=JSON_PATH)

    testAndRecord(model, object, NUM_CLASSES, TEST_BATCH_SIZE, TEST_NUM_WORKERS, subTask, fiveFoldName, MODEL_NAME, BASE_DATASET_NAME)
    # except Exception as e:# 写一个except
    #     print(f"发生异常，信息是: {e}")

if __name__ == '__main__':
    main()