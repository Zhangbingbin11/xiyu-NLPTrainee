# Author:Zhangbingbin 
# Time:2019/7/9
# 6.1 赎金信
# 给定一个ransom字符串 和 一个magazine字符串，判断第一个字符串能不能由第二个字符串里面的字符构成

#思路：按照ransom中的字符 去用空字符 替换magazine 中的字符 后面根据替换后的magazine字符串的长度和替换前的变化 来判断TRUE OR FALSE
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        for r in ransomNote:
            current_mag = magazine.replace(r,"",1)
            if len(current_mag) == len(magazine):
                return False
            magazine =current_mag
        return True

# 6.2字符串转化整数（atoi）
# 将字符串转化为整数，从第一个非空格字符开始
#(正则化)
import re
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        return max(min(int(*re.findall('^[\+\-]?\d+', str.lstrip())), 2**31 - 1), -2**31)
res = Solution().myAtoi("abs")
print(res)

# 6.3去除标签
# 去除以下html文件中的标签，只显示文本信息
import re
html = "<div>" \
       "<p>岗位职责：</p>"\
       "<p>1、一年以上经验"\
       "<p>&nbsp;</p><br>"\
       "<div>"
pattern = re.compile(r'<[^>]+>',re.M)  #('<[^>]+>'表示以 < 开头，中间是 非> 以外的任意字符，以 > 结束的字符串)
new_html = pattern.sub('', html)
print(new_html)

# 6.4 Pandas基础
#1.series A 以series B为分组依据，然后计算分组后的平均值
#2.如何创建一个以'2000-01-02'开始包含10 个周六的TimeSeries
import numpy as np
import pandas as pd

## a1 = np.random.choice(a=5, size=3, replace=False, p=None)
## 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
## replace 代表的意思是抽样之后还放不放回去，如果是False的话，则不放回，True则是有放回,默认是有放回

fruit = pd.Series(np.random.choice(['apple','banana','carrot'],10))
weights = pd.Series(np.linspace(1,10,10))  #在指定的间隔内（1，10）返回均匀间隔的10个数字 [1,2,3,4,5,6,7,8,9,10]
print(weights.groupby(fruit).mean())

#2.如何创建一个以'2000-01-02'开始包含10 个周六的TimeSeries

##语法：pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)
##该函数主要用于生成一个固定频率的时间索引，在调用构造方法时，必须指定start、end、periods中的两个参数值，否则报错。
ans = pd.Series(np.random.randint(1,10,10),pd.date_range('2000-01-02',periods=10,freq='W-SAT'))
print(ans)  ##输出的第二列数字是什么意思？？？？