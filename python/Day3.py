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
    
# 3.3 交错字符串
# 给定三个字符串s1,s2,s3，验证s3是否是由s1和s2交错组成的。
# 思路：（动态规划）
# dp[i][j] 表示用 s1的前 (i+1)和 s2 的前 (j+1) 个字符，总共 (i+j+2)个字符，是否交错构成 s3的前缀。
# 为了求出 dp[i][j]，我们需要考虑 2 种情况：
# 1.当s1 的第 i个字符和 s2的第 j 个字符都不能匹配 s3的第 k个字符，其中 k=i+j 。
# 这种情况下，s1 和 s2 的前缀无法交错形成 s3 长度为 k 的前缀。因此，我们让 dp[i][j] 为 False。
# 2.当 s1 的第 i个字符或者 s2 的第 j 个字符可以匹配 s3的第 k 个字符，其中 k=i+j 。
# 假设匹配的字符是 x且与 s1的第 i 个字符匹配，我们就需要把 x放在已经形成的交错字符串的最后一个位置。
# 此时，为了我们必须确保 s1 的前 (i-1) 个字符和 s2 的前 j个字符能形成 s3的一个前缀。
# 类似的，如果我们将 s2的第 j个字符与 s3的第 k 个字符匹配，我们需要确保 s1的前 i个字符和 s2的前 (j-1)个字符能形成 s3 的一个前缀，
# 我们就让 dp[i][j] 为 True。
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1)+len(s2)!=len(s3):
            return False
        dp=[[False for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        dp[0][0]=True
        for i in range(1, len(s1)+1):
            dp[i][0] = dp[i-1][0] and s3[i-1]==s1[i-1]  # 边界：之前的需要符合 并且s1和s3对应位置相同
        for i in range(1, len(s2)+1):
            dp[0][i] = dp[0][i-1] and s3[i-1]==s2[i-1]  # 边界：之前的需要符合 并且s1和s3对应位置相同
        for i in range(1, len(s1)+1):
            for j in range(1, len(s2)+1):
                # print i, j, s1[i-1], s3[i+j-1], s2[j-1], s3[i+j-1]
                #          s1的第 i 个字符与s3的第 k 个匹配    或者   s2的第 j 个字符与s3的第 k 个匹配
                dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])
        return dp[-1][-1]
res = Solution().isInterleave("abc","def","adebfc")
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
