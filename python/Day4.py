# Author:Zhangbingbin 
# Time:2019/7/4
# 4.1 重复的子字符串
# 给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成，给定的字符串只含有小写英文字母，并且长度不超过10000.
# 思路：子串的长度一定是小于等于字符串长度的一半：
#      从 第二个字符开始依次与第一个字符对比，如果相等且len(s)%i==0 且 s[:i]*len(s)//i==s 则返回True
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) < 2:
            return False
        mid = len(s)//2
        i = 1
        while i <= mid:
            if s[i] == s[0]:
                if len(s) % i == 0:
                    substr = s[:i]
                    if s == substr*(len(s)//i):
                        return True
            i += 1
        return False
res =Solution().repeatedSubstringPattern("abcabcabc")
print(res)

#4.2 Z字形变换
#将一个给定字符串根据给定的行数，以从上往下、从左到右进行Z字形排列
#思路：两种状态，一种垂直向下，还有一种斜向上
class Solution():
    def convert(self, s: str, numRows: int) -> str:
        if not s:
            return ""
        if numRows == 1:return s
        s_Rows = [""] * numRows
        i  = 0
        n = len(s)
        while i < n:
            for j in range(numRows):
                if i < n:
                    s_Rows[j] += s[i]
                    i += 1
            for j in range(numRows-2,0,-1):
                if i < n:
                    s_Rows[j] += s[i]
                    i += 1
        return "".join(s_Rows)
ans = Solution().convert("leetcode",3)
print(ans)

# 4.3 不同的子序列
#给定一个字符串S 和一个字符串T ,计算在S的子序列中 T出现的个数。
# 一个字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。
#（例如：“ACE”是“ABCDE”的一个子序列，而“AEC”不是）
# 思路：（动态规划）定义当t为空字符串时dp=0
# 状态转移方程：   if t[i - 1] == s[j - 1]:
#                     dp[i][j] = dp[i - 1][j - 1]  + dp[i][j - 1]
#                else:
#                     dp[i][j] = dp[i][j - 1]
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        dp = [[0 for i in range(len(s)+1)] for j in range(len(t) + 1)]
        for j in range(len(s) + 1):
            dp[0][j] = 1
        for i in range(1, len(t) + 1):
            for j in range(1, len(s) + 1):
                if t[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1]
ans = Solution().numDistinct("abccc","abc")
print(ans)

# 4.4 翻转单词顺序
# #输入一个英文句子，翻转句子中单词的顺序，而单词中的字母顺序不变。例如：“I am a student.”-->"student. a am I"
# #思路：按空格切分 然后翻转 再连接
class Solution():
    def ReverseSentence(self, s):
        #         # write code here
        if s is None or len(s) == 0:
            return s
        return " ".join(s.split(' ')[::-1])
ans = Solution().ReverseSentence("I am a student.")
print(ans)

# 4.5 Pandas基础
#Series将一日期字符串 转化 为时间
#Series从时间序列中提取年/月/日/时/分/秒
import pandas as pd
ser = pd.Series(['01 Jan 2019',
                 '01-01-2019',
                 '20190101',
                 '2019/01/01',
                 '2019-01-01T12:20:59'])
date = pd.to_datetime(ser) #Series将一日期字符串 转化 为时间
year = date.dt.year
month = date.dt.month
day = date.dt.day
hour = date.dt.hour
minute = date.dt.minute
second = date.dt.second
print(date)
print(year)
print(month)
print(day)
print(hour)
print(minute)
print(second)

