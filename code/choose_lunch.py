lunchs = {
    1:'干净的食堂',
    2:'老地方',
    3:'土豆粉',
    4:'烤肉拌饭',
    5:'一间面馆',
    6:'兰州拉面',
    7:'砂锅米线',
    8:'拿度尼',
    9:'黄焖鸡',
    10:'馄饨',
    11:'烧肉卷饼',
    12:'犟骨头',
    13:'韩食屋',
    14:'羊肉汤',
    15:'板面',
    16:'佰饺汇',
    17:'粥铺快餐',
    18:'牛肉汤',
    19:'金年任',
    20:'米小福快餐',
    21:'翻滚吧牛宝宝',
    22:'南京小笼包',
    23:'超意兴',
    24:'创新谷食堂',
    25:'伊尊阁牛肉面',
    26:'骨汤麻辣烫',
    27:'麻辣拌',
    # 28:'拿度尼',
    # 29:'黄焖鸡',
    # 30:'馄饨',
    }

import random
i = 0
# print(list(lunchs.keys()))
while True:

    if i >=3:
        print(f'你已经随机了{i+1}次了，建议选择拿度尼！')
        break
    if i ==1:
        index = random.randint(1,list(lunchs.keys())[-1]*2)
        if index > 27:
            print("今天午餐的选择是：拿度尼")
        else:
            print('今天午餐的选择是：',lunchs[index])

    if i ==2:
        index = random.randint(1,list(lunchs.keys())[-1]*5)
        if index > 27:
            print("今天午餐的选择是：拿度尼")
        else:
            print('今天午餐的选择是：',lunchs[index])
    
    if i==0:
        index = random.randint(1,27)
        print('今天午餐的选择是：',lunchs[index])
        print('请问您是否选择再次随机？但是这会导致拿度尼被选中的概率达到50%')
    if i==1:
        print('请问您是否选择再次随机？但是这会导致拿度尼被选中的概率达到80%')
    if i==2:
        print('请问您是否选择再次随机？但是这会导致拿度尼被选中的概率达到100%')
    willing = input()
    
    if willing =='ok':
        break
    
    i+=1

