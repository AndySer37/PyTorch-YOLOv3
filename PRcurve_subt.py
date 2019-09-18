from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--config_path", type=str, default="config/yolov3_subt.cfg", help="path to model config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_150.pth", help="path to weights file")
    parser.add_argument("--data_config_path", type=str, default="config/subt.data", help="path to data config file")
    parser.add_argument("--class_path", type=str, default="data/subt.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.99, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    if cuda:
        model.cuda()

    model.eval()  # Set in evaluation mode
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config["valid"]
    print (test_path)
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    iou_thres = 0.5
    conf_thres = 0.99

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs_list = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    # non_list = [1, 2, 3, 7, 9, 10, 11, 13, 14, 16, 18, 19, 19, 21, 24, 26, 27, 28, 29, 29, 30, 36, 39, 41, 42, 45, 48, 48, 49, 50, 55, 58, 59, 60, 63, 64, 65, 66, 70, 71, 72, 74, 75, 77, 78, 80, 82, 83, 83, 85, 89, 90, 95, 96, 98, 99, 100, 102, 102, 103, 103, 104, 106, 107, 108, 109, 110, 115, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 131, 132, 132, 133, 134, 134, 137, 142, 145, 146, 148, 156, 156, 156, 158, 158, 158, 159, 160, 162, 164, 166, 167, 168, 169, 171, 173, 176, 177, 180, 181, 181, 182, 183, 185, 186, 187, 189, 192, 194, 195, 198, 201, 205, 207, 208, 209, 211, 212, 214, 214, 218, 219, 220, 227, 230, 231, 233, 234, 235, 240, 242, 244, 245, 247, 248, 249, 251, 252, 254, 254, 255, 256, 259, 264, 265, 266, 266, 272, 272, 274, 275, 276, 280, 281, 282, 282, 282, 285, 286, 287, 288, 292, 294, 298, 300, 302, 303, 308, 310, 312, 318, 319, 321, 324, 326, 328, 329, 330, 331, 332, 333, 334, 334, 335, 339, 342, 347, 350, 353, 356, 359, 361, 361, 362, 364, 366, 367, 367, 368, 370, 371, 373, 379, 381, 382, 383, 383, 384, 385, 389, 391, 392, 393, 395, 396, 397, 399, 401, 403, 405, 406, 407, 407, 409, 411, 413, 415, 416, 417, 418, 419, 420, 420, 423, 423, 425, 430, 433, 436, 437, 438, 439, 440, 442, 443, 444, 447, 450, 451, 451, 454, 456, 458, 459, 462, 462, 463, 464, 465, 466, 472, 475, 475, 477, 479, 481, 485, 486, 487, 488, 490, 491, 492, 495, 496, 499, 501, 502, 503, 505, 505, 506, 509, 510, 512, 513, 514, 516, 519, 522, 523, 524, 525, 527, 528, 529, 531, 536, 537, 539, 540, 540, 542, 550, 551, 555, 557, 558, 559, 560, 563, 564, 565, 566, 568, 569, 570, 572, 573, 573, 575, 578, 584, 585, 586, 587, 588, 593, 594, 595, 598, 602, 603, 604, 606, 607, 609, 611, 612, 613, 614, 615, 616, 618, 619, 620, 622, 623, 624, 627, 628, 629, 630, 631, 634, 634, 636, 641, 642, 644, 645, 646, 647, 649, 651, 653, 654, 655, 657, 658, 661, 664, 665, 667, 668, 668, 670, 672, 675, 680, 682, 683, 684, 686, 687, 688, 689, 690, 691, 693, 694, 696, 698, 699, 703, 705, 709, 710, 714, 716, 718, 720, 721, 722, 723, 725, 726, 728, 729, 730, 737, 738, 739, 741, 743, 744, 746, 750, 750, 754, 755, 757, 758, 759, 760, 762, 764, 765, 766, 769, 770, 772, 772, 773, 775, 777, 778, 779, 780, 785, 788, 791, 794, 795, 796, 797, 798, 799, 800, 802, 804, 808, 808, 809, 811, 812, 814, 818, 819, 819, 820, 826, 826, 827, 828, 829, 832, 833, 835, 836, 837, 838, 842, 842, 843, 845, 847, 849, 850, 851, 852, 854, 856, 857, 857, 859, 860, 864, 866, 866, 868, 869, 874, 875, 876, 880, 882, 884, 889, 890, 892, 895, 896, 897, 898, 899, 900, 902, 904, 904, 907, 908, 909, 911, 913, 915, 918, 920, 922, 922, 923, 925, 927, 928, 929, 930, 932, 933, 934, 937, 939, 942, 946, 947, 948, 951, 952, 956, 957, 958, 959, 961, 966, 967, 969, 970, 971, 973, 974, 977, 978, 979, 985, 986, 988, 990, 991, 995, 996, 1000, 1001, 1004, 1005, 1006, 1007, 1008, 1013, 1015, 1016, 1017, 1018, 1019, 1021, 1024, 1029, 1031, 1032, 1034, 1036, 1037, 1039, 1043, 1044, 1045, 1046, 1047, 1049, 1053, 1056, 1059, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1072, 1073, 1074, 1075, 1077, 1078, 1078, 1079, 1081, 1082, 1083, 1084, 1085, 1086, 1088, 1090, 1096, 1097, 1097, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1108, 1111, 1114, 1116, 1116, 1117, 1118, 1118, 1121, 1122, 1126, 1127, 1129, 1131, 1133, 1134, 1137, 1142, 1147, 1148, 1150, 1150, 1151, 1152, 1154, 1155, 1156, 1157, 1157, 1158, 1161, 1162, 1163, 1164, 1165, 1166, 1168, 1170, 1171, 1173, 1176, 1178, 1179, 1181, 1184, 1185, 1186, 1187, 1189, 1192, 1194, 1195, 1198, 1200, 1200, 1202, 1203, 1205, 1206, 1209, 1210, 1212, 1213, 1213, 1215, 1216, 1217, 1219, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1238, 1238, 1240, 1240, 1242, 1244, 1250, 1253, 1253, 1254, 1255, 1256, 1258, 1259, 1262, 1262, 1263, 1265, 1267, 1268, 1269, 1271, 1272, 1275, 1276, 1277, 1279, 1280, 1281, 1282, 1283, 1284, 1287, 1289, 1290, 1293, 1294, 1295, 1295, 1296, 1298, 1300, 1302, 1303, 1305, 1307, 1310, 1311, 1316, 1318, 1320, 1322, 1324, 1325, 1326, 1329, 1332, 1334, 1337, 1340, 1341, 1341, 1342, 1346, 1347, 1349, 1350, 1352, 1356, 1357, 1358, 1361, 1362, 1364, 1365, 1366, 1368, 1369, 1374, 1375, 1375, 1377, 1380, 1382, 1384, 1386, 1388, 1389, 1390, 1392, 1394, 1394, 1395, 1396, 1398, 1400, 1402, 1408, 1409, 1410, 1412, 1413, 1414, 1416, 1416, 1418, 1419, 1421, 1422, 1422, 1423, 1423, 1424, 1425, 1427, 1430, 1431, 1432, 1434, 1436, 1439, 1440, 1446, 1448, 1450, 1451, 1452, 1453, 1455, 1457, 1458, 1458, 1460, 1462, 1463, 1464, 1464, 1466, 1468, 1469, 1470, 1470, 1472, 1473, 1479, 1480, 1480, 1484, 1486, 1487, 1489, 1498, 1499, 1502, 1504, 1505, 1506, 1508, 1511, 1512, 1514, 1519, 1521, 1521, 1522, 1523, 1525, 1528, 1530, 1531, 1534, 1535, 1538, 1539, 1542, 1545, 1548, 1550, 1550, 1554, 1554, 1559, 1566, 1567, 1573, 1575, 1576, 1577, 1580, 1581, 1582, 1584, 1585, 1586, 1590, 1591, 1592, 1593, 1593, 1594, 1595, 1596, 1598, 1600, 1601, 1602, 1605, 1606, 1607, 1609, 1609, 1610, 1610, 1610, 1611, 1612, 1613, 1616, 1621, 1622, 1622, 1625, 1626, 1630, 1631, 1632, 1637, 1640, 1643, 1644, 1647, 1647, 1648, 1649, 1651, 1654, 1655, 1656, 1657, 1659, 1661, 1662, 1663, 1666, 1666, 1670, 1673, 1676, 1680, 1681, 1683, 1684, 1689, 1691, 1695, 1698, 1701, 1703, 1703, 1712, 1713, 1714, 1715, 1716, 1718, 1719, 1721, 1723, 1724, 1727, 1728, 1735, 1739, 1740, 1741, 1742, 1742, 1748, 1750, 1751, 1752, 1754, 1756, 1760, 1760, 1762, 1763, 1766, 1767, 1767, 1769, 1772, 1778, 1779, 1781, 1784, 1784, 1786, 1788, 1791, 1793, 1794, 1797, 1801, 1805, 1807, 1808, 1809, 1813, 1815, 1816, 1818, 1821, 1823, 1824, 1826, 1828, 1830, 1831, 1831, 1833, 1841, 1843, 1845, 1848, 1851, 1852, 1853, 1856, 1857, 1858, 1860, 1861, 1863, 1864, 1865, 1866, 1867, 1869, 1871, 1872, 1873, 1875, 1878, 1882, 1884, 1885, 1889, 1890, 1892, 1894, 1895, 1897, 1899, 1899, 1899, 1900, 1903, 1903, 1904, 1906, 1907, 1911, 1915, 1916, 1918, 1919, 1921, 1923, 1925, 1928, 1930, 1931, 1933, 1934, 1934, 1935, 1936, 1937, 1939, 1940, 1941, 1944, 1945, 1946, 1950, 1953, 1955, 1957, 1958, 1959, 1959, 1959]
    count = 0
    result_list = [0,0,0,0] # TP TN FP FN
    all_data = 0
    print("\nPerforming object detection:")
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        count += 1
        # if count not in non_list:
        # Configure input
        imgs = Variable(imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(imgs)
            # print (detections)
            detections = non_max_suppression(detections, conf_thres, opt.nms_thres)
            print (detections)

        # Save image and detections
        # imgs_list.extend(imgs)
        # img_detections.extend(detections)

        for detect_i, (labels, detection, img) in enumerate(zip(targets, detections, imgs)):
            
            for label in labels:
                _bool = False
                # print (label)
                mask_label = np.zeros(img.shape[1:3], np.float64)
                mask_label[int(label[2]):int(label[4]),int(label[3]):int(label[5])] = 1
                all_data += 1
                if detection is not None:
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        mask_predict = np.zeros(img.shape[1:3], np.float64)
                        mask_predict[int(x1):int(x2),int(y1):int(y2)] = 1

                        _and = sum(sum(np.logical_and(mask_label, mask_predict)))
                        _or = sum(sum(np.logical_or(mask_label, mask_predict)))
                        _iou = float(_and/_or)

                        if conf >= conf_thres :
                            if cls_pred == label[1] and not _bool and _iou >= iou_thres:
                                _bool = True
                                result_list[0] += 1
                            elif cls_pred != label[1] or _iou < iou_thres:
                                result_list[2] += 1
                
                if not _bool:
                    # non_list.append(count)
                    result_list[1] += 1
    print (result_list)
    print (all_data)
    print ("Recall : ", float(result_list[0]/all_data))
    print ("Precision : ", float(result_list[0]/(result_list[0] + result_list[2])))
    # print(non_list)
