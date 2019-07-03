# Author:Zhangbingbin 
# Time:2019/7/3
#3.1最长的公共前缀
#编写一个函数来查找字符串数组中的最长公共前缀，如果不存在公共前缀，返回空字符串“ ”。

# 思路：最长的前缀不会超过最短的字符串，那么可以遍历最短的字符串的长度。
# 1.找出长度最短的字符串长度；
# 2.依次与前一个字符串比较。
class Solution():
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        minl = min([len(x) for x in strs])
        end = 0 #定义一个初始索引位置（比较每个字符串 0 索引处的值）
        while end < minl:  #索引要小于 最小字符串长度 才往下走
            for i in range(1,len(strs)):
                if strs[i][end]!= strs[i-1][end]: #比较相邻两个字符串 end索引位置元素 若不相等则返回 end索引之前的字母 若相等则end+1继续往后比较
                    return strs[0][:end]
            end += 1
        return strs[0][:end]
if __name__ == "__main__":
    res = Solution().longestCommonPrefix(["flower","flow","flight"])
    print(res)

# 3.2 电话号码的字母组合
# 给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。数字到字母的映射与电话键盘相同。
# 如输入：“23”
# 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
# 
# 递归
# 思路：1.如果只输入一个数字digit,则返回对应数字下的字母letter列表，如: 输入：2;输出：["a","b","c"]
#     2.如果输入的是多个digits,则将 数字分为左右两部分 left是digits[:-1] right是digits[-1]
#     3.这里用到 递归 left 继续调用这个函数
class Solution():
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        letters = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        res = []
        if len(digits) == 1:
            return list(letters[int(digits[0])-2])
        #将数字串分为左右部分
        left = self.letterCombinations(digits[:-1]) #left是digits[:-1]
        right = list(letters[int(digits[-1])-2]) #right是digits[-1]对应的letters列表
        for i in left:
            for j in right:
                res.append(i+j) #拼接
        return res
if __name__ == "__main__":
    res = Solution().letterCombinations("25")
    print(res)

#3.4 第一个只出现一次的字符
#在一个字符串（1<=字符串长度<=10000,全部由字母组成）中找到第一个只出现一次的字符，并返回他的位置
#思路：利用Counter函数给字符串中的字符计数，按字符串顺序遍历，取出 count==1 的字符
from collections import Counter
class Solution():
    def FirstNotRepeatingChar(self, s):
        if not s:
            return -1
        count = Counter(s)
        for i,c in enumerate(s): # i 是索引，c 是 s 中的字符
            if count[c] == 1:
                return i,c
ansi,ansc = Solution().FirstNotRepeatingChar("abjddundsiassbuenjs")
print(ansi,ansc)

#3.5 Pandas基础
#series中计数排名前2的元素
#如何将数字系列分成10个相同大小的组
#如何将系列中每个元素的第一个字符转化为大写
import pandas as pd
import numpy as np
ser = pd.Series(["a","a","b","c","d","c","c"])
count = ser.value_counts()
print(count)
count_2 = ser.value_counts().index[:2]#series中计数排名前2的元素
print(count_2)

ser1 = pd.Series(np.random.random(10))#生成0~1的随机浮点数
print(ser1.head())
groups = pd.qcut(ser1,q=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])#将数字系列分成大小相同的10个组
print(groups)

ser2 = pd.Series(["how","to","go"])
print(ser2.map(lambda x:x.title())) #将系列中每个元素的第一个字符转化为大写
