## Decision Tree Learning Algorithm

### Results

Dataset-1: Number of training examples (m): 10 with features: 3 (Deterministic Dataset)

**Decision Tree-1**:
- Depth 0, Root: Split on feature: 2
    - Depth 1, Left: Split on feature: 0
        - Left leaf node with indices [0, 1, 4, 7]
        - Right leaf node with indices [5]
    - Depth 1, Right: Split on feature: 1
        - Left leaf node with indices [8]
        - Right leaf node with indices [2, 3, 6, 9]

Dataset-2: Number of training examples (m): 1000 with features: 7 (Randomized Dataset)

**Decision Tree-2**:
- Depth 0, Root: Split on feature: 3
    - Depth 1, Left: Split on feature: 6
        - Depth 2, Left: Split on feature: 2
            - Depth 3, Left: Split on feature: 4
                - Depth 4, Left: Split on feature: 1
                    - Left leaf node with indices [17, 31, 67, 140, 174, 186, 199, 217, 226, 252, 254, 256, 313, 372, 412, 456, 490, 529, 552, 573, 620, 625, 674, 722, 753, 790, 801, 826, 851, 897, 900, 967, 984]
                    - Right leaf node with indices [26, 39, 80, 107, 121, 225, 258, 300, 306, 328, 343, 480, 482, 485, 567, 574, 608, 636, 661, 701, 711, 742, 800, 823, 838, 854, 862, 876, 891, 893, 972]
                - Depth 4, Right: Split on feature: 5
                    - Left leaf node with indices [109, 142, 194, 270, 329, 428, 437, 443, 445, 452, 472, 579, 581, 645, 693, 697, 747, 784, 792, 843, 858, 892, 903, 951, 955]
                    - Right leaf node with indices [22, 25, 146, 189, 302, 360, 385, 463, 518, 643, 684, 710, 760, 785, 803, 825, 915, 932, 942, 979]
            - Depth 3, Right: Split on feature: 4
                - Depth 4, Left: Split on feature: 5
                    - Left leaf node with indices [1, 2, 42, 169, 180, 195, 260, 297, 339, 374, 441, 469, 497, 648, 678, 696, 718, 821, 866, 869, 890, 895, 938, 980]
                    - Right leaf node with indices [115, 131, 135, 138, 145, 149, 207, 263, 282, 295, 322, 325, 332, 347, 381, 383, 394, 406, 426, 447, 508, 509, 533, 541, 589, 591, 623, 689, 698, 716, 719, 772, 798, 802, 832, 847, 864, 873, 987]
                - Depth 4, Right: Split on feature: 1
                    - Left leaf node with indices [49, 64, 88, 119, 139, 154, 156, 164, 197, 210, 218, 229, 242, 276, 291, 293, 310, 380, 411, 429, 431, 466, 475, 502, 536, 549, 564, 613, 706, 734, 759, 769, 846, 883, 957, 973, 976, 991, 996]
                    - Right leaf node with indices [29, 102, 114, 120, 148, 153, 159, 177, 214, 255, 274, 299, 303, 314, 331, 348, 368, 397, 465, 478, 507, 526, 565, 600, 628, 637, 656, 725, 743, 766, 779, 780, 815, 877, 960, 961]
        - Depth 2, Right: Split on feature: 0
            - Depth 3, Left: Split on feature: 1
                - Depth 4, Left: Split on feature: 4
                    - Left leaf node with indices [5, 89, 173, 188, 215, 232, 272, 285, 326, 357, 359, 375, 457, 483, 588, 658, 673, 761, 778, 791, 808, 959, 998]
                    - Right leaf node with indices [0, 55, 81, 151, 185, 212, 235, 237, 241, 284, 373, 408, 417, 450, 451, 597, 642, 666, 671, 726, 844, 860, 871, 878, 965, 969]
                - Depth 4, Right: Split on feature: 4
                    - Left leaf node with indices [10, 20, 50, 77, 96, 123, 152, 155, 205, 221, 271, 279, 304, 321, 341, 349, 376, 489, 514, 523, 527, 535, 538, 553, 592, 593, 596, 607, 629, 646, 652, 730, 788, 809, 875, 907, 917, 929]
                    - Right leaf node with indices [4, 6, 82, 83, 93, 162, 172, 196, 213, 223, 275, 283, 384, 442, 444, 511, 539, 543, 618, 733, 736, 746, 786, 841, 842, 859, 888, 913, 933, 936, 941, 952, 995]
            - Depth 3, Right: Split on feature: 4
                - Depth 4, Left: Split on feature: 1
                    - Left leaf node with indices [19, 23, 65, 73, 84, 91, 95, 113, 315, 318, 354, 364, 389, 413, 419, 476, 501, 515, 521, 528, 544, 601, 616, 622, 627, 692, 694, 695, 751, 756, 767, 799, 829, 925, 966, 983]
                    - Right leaf node with indices [45, 90, 92, 125, 147, 184, 219, 278, 324, 333, 344, 358, 462, 470, 520, 572, 576, 635, 653, 740, 765, 771, 852, 857, 914, 930, 943, 968]
                - Depth 4, Right: Split on feature: 1
                    - Left leaf node with indices [7, 56, 94, 239, 257, 281, 288, 311, 346, 370, 377, 424, 468, 473, 487, 493, 524, 542, 566, 655, 712, 819, 840, 905, 918, 964, 982]
                    - Right leaf node with indices [9, 87, 193, 224, 265, 367, 393, 403, 421, 459, 479, 486, 559, 590, 632, 644, 662, 702, 708, 713, 737, 810, 867, 906, 939, 990]
    - Depth 1, Right: Split on feature: 6
        - Depth 2, Left: Split on feature: 2
            - Depth 3, Left: Split on feature: 5
                - Depth 4, Left: Split on feature: 0
                    - Left leaf node with indices [16, 36, 52, 103, 157, 160, 209, 259, 309, 312, 363, 365, 382, 386, 471, 494, 510, 537, 554, 558, 603, 614, 639, 640, 654, 727, 782, 793, 804, 806, 820, 901, 940, 974, 986]
                    - Right leaf node with indices [47, 53, 54, 69, 71, 106, 111, 168, 181, 245, 371, 391, 398, 449, 512, 649, 668, 676, 724, 752, 822, 839, 881, 884, 904, 924, 970]
                - Depth 4, Right: Split on feature: 0
                    - Left leaf node with indices [3, 15, 21, 24, 44, 86, 98, 105, 132, 141, 250, 251, 262, 267, 323, 330, 350, 362, 423, 464, 540, 548, 556, 604, 609, 633, 686, 717, 748, 777, 789, 824, 885, 923, 937, 985, 989]
                    - Right leaf node with indices [34, 40, 118, 163, 190, 211, 227, 236, 292, 337, 351, 499, 522, 585, 634, 651, 667, 675, 680, 721, 755, 774, 794, 831, 863, 889]
            - Depth 3, Right: Split on feature: 1
                - Depth 4, Left: Split on feature: 0
                    - Left leaf node with indices [60, 63, 79, 122, 170, 179, 187, 234, 290, 320, 327, 388, 405, 546, 568, 619, 621, 638, 660, 685, 700, 728, 757, 796, 850, 853, 916, 953, 999]
                    - Right leaf node with indices [35, 61, 66, 99, 116, 166, 206, 220, 222, 269, 305, 334, 361, 379, 430, 439, 461, 477, 488, 492, 505, 513, 519, 545, 577, 587, 602, 650, 659, 679, 720, 739, 744, 758, 763, 812, 813, 845, 880, 882, 896, 899, 902, 912, 949, 988]
                - Depth 4, Right: Split on feature: 0
                    - Left leaf node with indices [8, 76, 117, 167, 191, 230, 261, 273, 294, 296, 308, 402, 414, 427, 467, 562, 582, 615, 715, 729, 738, 749, 830, 836, 911, 921, 922, 934]
                    - Right leaf node with indices [11, 37, 59, 97, 112, 136, 171, 203, 231, 264, 266, 342, 345, 454, 498, 516, 583, 612, 626, 669, 677, 681, 683, 691, 705, 731, 750, 781, 807, 828, 834, 874, 908, 920, 931, 945, 992]
        - Depth 2, Right: Split on feature: 2
            - Depth 3, Left: Split on feature: 1
                - Depth 4, Left: Split on feature: 4
                    - Left leaf node with indices [32, 70, 78, 126, 178, 182, 201, 208, 216, 228, 248, 277, 280, 307, 353, 355, 420, 422, 434, 436, 503, 525, 580, 599, 606, 641, 723, 754, 762, 773, 776, 783, 827, 870, 919, 956]
                    - Right leaf node with indices [68, 72, 74, 75, 129, 144, 240, 249, 316, 369, 387, 446, 448, 458, 484, 495, 547, 550, 861, 865, 910, 935, 944, 946, 971]
                - Depth 4, Right: Split on feature: 4
                    - Left leaf node with indices [12, 14, 18, 51, 57, 62, 101, 124, 130, 143, 165, 175, 268, 286, 336, 356, 378, 390, 399, 409, 425, 481, 491, 500, 586, 598, 610, 670, 672, 699, 703, 707, 732, 770, 795, 817, 837, 947, 954]
                    - Right leaf node with indices [13, 30, 41, 48, 58, 104, 137, 161, 246, 253, 298, 366, 392, 395, 407, 438, 440, 455, 460, 517, 532, 551, 569, 584, 624, 657, 687, 764, 787, 797, 811, 872, 894, 909, 926, 928, 962, 963, 975, 981]
            - Depth 3, Right: Split on feature: 0
                - Depth 4, Left: Split on feature: 4
                    - Left leaf node with indices [85, 110, 127, 128, 183, 243, 317, 352, 401, 404, 416, 432, 433, 435, 496, 561, 563, 571, 630, 663, 664, 704, 709, 768, 848, 855, 958, 978]
                    - Right leaf node with indices [43, 158, 192, 198, 204, 233, 289, 301, 319, 335, 338, 418, 530, 555, 631, 665, 682, 775, 805, 835, 856, 868, 879, 927, 997]
                - Depth 4, Right: Split on feature: 4
                    - Left leaf node with indices [27, 38, 46, 108, 133, 150, 176, 200, 238, 396, 400, 474, 506, 575, 578, 595, 605, 690, 741, 745, 818, 833, 849, 887, 898, 948, 950, 977, 994]
                    - Right leaf node with indices [28, 33, 100, 134, 202, 244, 247, 287, 340, 410, 415, 453, 504, 531, 534, 557, 560, 570, 594, 611, 617, 647, 688, 714, 735, 814, 816, 886, 993]

## Citation

Please note that the code and technical details made available are for anyone interested to learn. The repo is not open for collaboration.

If you happen to use the code from this repo, please cite my user name along with link to my profile: https://github.com/balarcode. Thank you!
