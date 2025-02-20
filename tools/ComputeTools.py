import os, numpy, torch
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F
# 这个函数主要用于找EER。
# 同时为了实际业务需求，提供了找出最接近目标假阴，假阳率的阈值，这个阈值被称为tuneThreshold
def tuneThresholdfromScore(scores, labels, target_fa=None, target_fr = None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    if target_fa:
        for tfa in target_fa:
            idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
            tunedThreshold.append([thresholds[idx], fpr[idx],fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr-fpr)))
    eer = max(fpr[idxE],fnr[idxE])*100
    return tunedThreshold, eer, fpr, fnr

# 该函数用于计算错误拒绝率， 错误接收率， 以及错误率。
# 给定一个scores列表， 以及对应的标签列表，所选择的阈值是scores列表中的每一个元素。
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

# 计算ASnorm得分
'''
args:
    score: 两个embedding的cosine得分
    embedding_enroll: 注册音频的embedding, shape:[1,192]
    embedding_test: 测试音频的embedding, shape: [1,192]
    embedding_cohort: 冒认语音集embedding, shape: [N,192]
    topk: 选取排名topk的样本进行计算mean与std。 
'''
def Compute_ASnorm(score, embedding_enroll, embedding_test, embedding_cohort, topk):
    score_1 = torch.matmul(embedding_cohort, embedding_enroll.T)[:,0]
    score_1 = torch.topk(score_1, topk, dim = 0)[0]
    mean_1 = torch.mean(score_1, dim = 0)
    std_1 = torch.std(score_1, dim = 0)
    score_2 = torch.matmul(embedding_cohort, embedding_test.T)[:,0]
    score_2 = torch.topk(score_2, topk, dim = 0)[0]
    mean_2 = torch.mean(score_2, dim = 0)
    std_2 = torch.std(score_2, dim = 0)

    score = 0.5 * (score - mean_1) / std_1 + 0.5 * (score - mean_2) / std_2
    return score