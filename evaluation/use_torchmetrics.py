import logging
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassSpecificity, MulticlassAUROC, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinarySpecificity, BinaryAUROC, BinaryF1Score, BinaryConfusionMatrix

# targets = torch.tensor([2, 1, 0, 0])
# preds = torch.tensor([[0.16, 0.26, 0.58],
#                 [0.22, 0.61, 0.17],
#                 [0.71, 0.09, 0.20],
#                 [0.05, 0.82, 0.13]])

class EvaClassify():

    def __init__(self, NUM_CLASSES):
    
        # 添加自己定义的模态
        # logging.basicConfig(filename=absolute_log_path, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.mutiClass_ACC = -1
        self.mutiClass_recall = -1
        self.mutiClass_specificity = -1
        self.mutiClass_f1 = -1

        self.ACC = -1
        self.SEN = -1
        self.SPE = -1
        self.AUC = -1
        self.F1_Score = -1

        self.num_classes = NUM_CLASSES
        if(NUM_CLASSES>2):
            # 指标ACC应该写为micro，就是数字加起来后，再统一做除法
            self.accuracy_method = MulticlassAccuracy(num_classes=NUM_CLASSES, average='micro').to(device)  # 先所有TP的加起来，再进行计算
            # 下面四个指标默认都是macro，就是各自求出后，再求平均数
            self.recall_method = MulticlassRecall(num_classes=NUM_CLASSES).to(device)
            self.specificity_method = MulticlassSpecificity(num_classes=NUM_CLASSES).to(device)
            self.auc_method = MulticlassAUROC(num_classes=self.num_classes).to(device)
            self.f1_method = MulticlassF1Score(num_classes=NUM_CLASSES).to(device)
            self.confusionMatrix_method = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(device)
        else:
            # 二分类不用考虑应该是macro还是micro，就不需要考虑宏观还是微观的问题
            self.accuracy_method = BinaryAccuracy().to(device)
            self.recall_method = BinaryRecall().to(device)
            self.specificity_method = BinarySpecificity().to(device)
            self.auc_method = BinaryAUROC().to(device)
            self.f1_method = BinaryF1Score().to(device)
            self.confusionMatrix_method = BinaryConfusionMatrix().to(device)

    def evaluateTwoClass(self, preds, targets):

        # # 创建相应的评估对象
        # accuracy = BinaryAccuracy().to(device)
        # recall = BinaryRecall().to(device)
        # specificity = BinarySpecificity().to(device)
        # auc = BinaryAUROC().to(device)
        # f1 = BinaryF1Score().to(device)
        # confusionMatrix = BinaryConfusionMatrix().to(device)

        # 计算指标：目前的指标可以理解为【每一类概率】的平均值（没有加权）
        ACC = self.accuracy_method(preds, targets)
        SEN = self.recall_method(preds, targets)
        SPE = self.specificity_method(preds, targets)
        AUC = self.auc_method(preds, targets)
        f1_score = self.f1_method(preds, targets)
        confusion = self.confusionMatrix_method(preds, targets)

        self.ACC = ACC.item()
        self.SEN = SEN.item()
        self.SPE = SPE.item()
        self.AUC = AUC.item()
        self.F1_Score = f1_score.item()

        logging.info(f'ACC: {ACC.item():.4f}')
        logging.info(f'SEN: {SEN.item():.4f}')
        logging.info(f'SPE: {SPE.item():.4f}')
        logging.info(f'AUC: {AUC.item():.4f}')
        logging.info(f'F1_Score: {f1_score.item():.4f}')
        logging.info(f'confusionMatrix：\n{confusion}')

    def evaluateMultiClass(self, preds, targets):

        # 计算指标：目前的指标可以理解为【每一类概率】的平均值（没有加权）
        self.ACC = self.accuracy_method(preds, targets).item()
        self.SEN = self.recall_method(preds, targets).item()
        ########################验证多分类的等价操作########################
        # # 找到目标类别的索引
        # SEN = 0
        # for current_idx in range(self.num_classes):
        #     class_idx = (targets == current_idx)
        #     # 找到预测为目标类别的样本
        #     true_positive = (preds.argmax(dim=1) == current_idx) & class_idx
        #     # 计算真正例的数量
        #     true_positive_count = true_positive.sum().item()
        #     # 计算目标类别的样本数量
        #     class_count = class_idx.sum().item()
        #     # 计算灵敏度
        #     SEN += (true_positive_count / class_count)
        # self.SEN = SEN / self.num_classes
        ################################################################
        self.SPE = self.specificity_method(preds, targets).item()
        self.AUC = self.auc_method(preds, targets).item()
        self.F1_Score = self.f1_method(preds, targets).item()
        confusion = self.confusionMatrix_method(preds, targets)

        # self.ACC = ACC.item()
        # self.SEN = SEN.item()
        # self.SPE = SPE.item()
        # self.AUC = AUC.item()
        # self.F1_Score = f1_score.item()

        logging.info(f'ACC: {self.ACC:.4f}')
        logging.info(f'SEN: {self.SEN:.4f}')
        logging.info(f'SPE: {self.SPE:.4f}')
        logging.info(f'AUC: {self.AUC:.4f}')
        logging.info(f'F1_Score: {self.F1_Score:.4f}')
        logging.info(f'confusionMatrix：\n{confusion}')

    def evaluate(self, preds, targets):

        if(self.num_classes>2):
            return self.evaluateMultiClass(preds, targets)
        else:
            return self.evaluateTwoClass(torch.argmax(preds, dim=-1), targets)

        

# 要区分准确率（针对所有分类）、查全率、查准率
# 添加最后一个为假反例 【和当前分类对比：真假针对target而言、正反针对pred而言】  

# targets = torch.tensor([2, 1, 0, 0, 1, 2, 1, 2, 1]).to(device)
# preds = torch.tensor([
#     [0.16, 0.26, 0.58],  # 对应类别2的预测概率最高
#     [0.22, 0.61, 0.17],  # 对应类别1的预测概率最高
#     [0.71, 0.09, 0.20],  # 对应类别0的预测概率最高
#     [0.05, 0.82, 0.13],  # 对应类别1的预测概率最高
#     [0.20, 0.78, 0.02],  # 对应类别1的预测概率最高
#     [0.5, 0.3, 0.2],     # 对应类别0的预测概率最高
#     [0.25, 0.45, 0.2],   # 对应类别1的预测概率最高
#     [0.2, 0.1, 0.7],     # 对应类别2的预测概率最高
#     [0.1, 0.8, 0.1],     # 对应类别2的预测概率最高
# ]).to(device)
# # 对应 2 1 0 1 1, 0 1 2 2
# # 对应 

# recoder = EvaClassify(NUM_CLASSES=3)
# recoder.evaluate(preds=preds, targets=targets)

# print(f'ACC: {recoder.ACC:.4f}')
# print(f'SEN: {recoder.SEN:.4f}')
# print(f'SPE: {recoder.SPE:.4f}')
# print(f'AUC: {recoder.AUC:.4f}')
# print(f'F1_Score: {recoder.F1_Score:.4f}')