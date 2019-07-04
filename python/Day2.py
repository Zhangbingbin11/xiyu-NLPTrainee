# 2.1最长回文子串
# 给定一个字符串s,找到s中最长的回文子串.你可以假设s的最大长度为1000.
# 思路：
# 1.如果字符串长度是小于2的，则s就是回文.
# 2.如果字符串头尾字符相等，但是中间部分不构成区间（最多只有 1 个元素）;
#  或者字符串头尾字符相等，且中间部分也是回文，则都返回True .此时回文长度为r-l+1
# 3.比较 取大值  赋值给result
class Solution():
    def longestPalindrome(self, s: str):
        size = len(s)
        if size <= 1:
            return s
        dp = [[False for _ in range(size)] for _ in range(size)]#列表表达式
        longest_l = 1
        res = s[0]
        # 因为只有 1 个字符的情况在最开始做了判断
        # 左边界一定要比右边界小，因此右边界从 1 开始
        for r in range(1, size):
            for l in range(r):
                if s[l] == s[r] and (r - l <= 2 or dp[l + 1][r - 1]):
                    dp[l][r] = True
                    cur_len = r - l + 1
                    if cur_len > longest_l:
                        longest_l = cur_len
                        res = s[l:r + 1]  #把回文赋值给res
        return res
a = Solution()
res = a.longestPalindrome("babad")
print(res)

#2.2回文子串
#给定一个字符串,计算这个字符串中有多少个回文子串.
#思路：（动态规划）
# 1.用动规记录和求出字符串s的所有是回文串的子字符串，然后用计数器counter技术
#  一样也是先从单个字符是回文的dp[i][i]= True开始记录
# 2.再到两个字符dp[i][i+1]  = s[i]==s[i+1]
# 3.再到后面的多个字符的回文dp[i][j] = (dp[i+1][j-1] and s[i+1]==s[j-1])
# 4.要注意遍历的方式
#  for j in range(1，n)
#      for i in range(j-1)
#  先从j开始遍历，代表以j结束的子串，然后i再从0开始去循环到j
class Solution():
    def countSubstrings(self, s):
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        counter = 0
        for i in range(n):
            dp[i][i] = True
            counter += 1
        for i in range(1, n):
            if s[i - 1] == s[i]:
                dp[i - 1][i] = True
                counter += 1
        for j in range(1, n):
            for i in range(j - 1):
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    counter += 1
        return counter
a = Solution()
c = a.countSubstrings("abababbb")
print(c)

#2.3分割回文串
#给定一个字符串s,将s分割成一些子串，使每个子串都是回文串。返回符合要求的最少分割次数。
#思路：（动态规划方法）
#1.如果字符串长度小于等于1 则分割次数为0
#2.定义一个初始的标记列表 （从-1 开始）（对应着索引位置字符还没有进入的时候的最大分割次数）
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <= 1:
            return 0
        # 定义初始标记列表（对应索引位置字符还没有进入的时候的最大分割次数）
        record = [index for index in range(-1, len(s))]
        for r in range(1, len(s)+1):
            for l in range(r):
                if s[l:r] == s[l:r][::-1]: #判断s[l:r]是否回文（反转相同）
                    record[r] = min(record[r], record[l]+1)
        return record[-1]


if __name__ == "__main__":
    s = "abaa"
    min_split = Solution().minCut(s)
    print(min_split)

#2.4字符串的排列
#输入一个字符串 按字典 打印出该字符串中字符的所有排列。例如输入字符串 abc,则打印出由a,b,c所能排列出来的所有字符串
# 字符串长度 不超过9（可能会有字符重复），只包括大小写字母
# 思路：将字符串分为两部分：第一个字符和剩下的字符 先排第一个字符的所有可能 再对应着依次排后面的字符 递推
def Permutation(s):
    if len(s) <= 0:
        return []
    res = []
    perm(s, res, '')       #初始化s_0 = '',s_0为首字符
    uniq = list(set(res))   #set()函数创建一个无需不重复的元素集，不需要判断元素是否重复
    return sorted(uniq)     #sorted()函数对任意对象进行排序

def perm(s, res, s_0):
    if s == '':
        res.append(s_0)
    else:
        for i in range(len(s)):
            perm(s[:i]+s[i+1:], res, s_0+s[i])  ###???

s = 'abc'
print(Permutation(s))

#2.5Pandas基础
#1.两个series的并集
#2.两个series的非共有元素
#3.如何获得series的最小值，25%分位数，中位数，75%分位数和最大值
import numpy as np
import pandas as pd
sA = pd.Series([1,2,3,4,5,6])
sB = pd.Series([5,6,7,8,9,10])
print(sA[~sA.isin(sB)])#元素在sA不在sB
u =pd.Series(np.union1d(sA,sB)) #sA∪sB
i =pd.Series(np.intersect1d(sA,sB)) #sA∩sB
print(u)
print(u[~u.isin(i)]) #非共有元素
print(np.percentile(sA,q=[0,25,50,75,100]))
