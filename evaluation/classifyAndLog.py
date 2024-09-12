import logging
from sklearn.metrics import accuracy_score, recall_score, specificity_score, f1_score, confusion_matrix

class EvaClassify():

    def __init__(self, num_classes):
    
        # 添加自己定义的模态
        # logging.basicConfig(filename=absolute_log_path, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.NUM_CLASSES = num_classes
        self.all_accuracy = -1
        self.all_recall = -1
        self.all_specificity = -1
        self.all_f1 = -1

    def evaluateByPredsAndTables(self, test_labels_cpu, test_preds_cpu):

        # 计算Accuracy，同时更新
        all_accuracy = accuracy_score(test_labels_cpu, test_preds_cpu)
        logging.info(f"Accuracy:{all_accuracy:.4f}")

        # 计算Recall（可以选择不同的average参数，macro、micro、weighted等）
        all_recall = recall_score(test_labels_cpu, test_preds_cpu, average='macro')
        logging.info(f"Recall:{all_recall:.4f}")

        # 计算Specificity（可以选择不同的average参数，macro、micro、weighted等）
        all_specificity = specificity_score(test_labels_cpu, test_preds_cpu, average='macro')
        logging.info(f"Specificity:{all_specificity:.4f}")

        # 计算F1 Score（可以选择不同的average参数，macro、micro、weighted等）
        all_f1 = f1_score(test_labels_cpu, test_preds_cpu, average='macro')
        logging.info(f"F1 Score:{all_f1:.4f}")


        ###############labels是需要分类的类别索引，如果不提供的话，就会自动从标签中提取至少出现一次的数值###################
        # 计算混淆矩阵
        confusion = confusion_matrix(test_labels_cpu, test_preds_cpu, labels=[i for i in range(self.NUM_CLASSES)])
        logging.info("Confusion Matrix:")
        logging.info(f"\n{confusion}")

        # 计算每一类的Recall
        recalls = []
        for i in range(len(confusion)):
            recall = recall_score(test_labels_cpu, test_preds_cpu, labels=[i], average=None)
            recalls.append(recall)

        # 计算每一类的Specificity
        specificitys = []
        for i in range(len(confusion)):
            specificity = specificity_score(test_labels_cpu, test_preds_cpu, labels=[i], average=None)
            specificitys.append(specificity)

        # 打印每一类的Recall和Specificity
        for i in range(len(confusion)):
            # 如果在字符串中使用{recalls[i]:.4f}，就会报错TypeError: unsupported format string passed to numpy.ndarray.__format__
            current_recall = float(recalls[i])
            current_specificity = float(specificitys[i])
            logging.info(f"Class {i} - Recall: {current_recall:.4f}, Specificity: {current_specificity:.4f}")

        self.all_accuracy = all_accuracy
        self.all_recall = all_recall
        self.all_specificity = all_specificity
        self.all_f1 = all_f1

    # def getAccuracy(self):
    #     return self.all_accuracy