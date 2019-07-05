#1.分割回文串
#思路：首先判断字符串长度是否<=1 是则输出0
#定义一个初始状态标记列表（第i个元素进来前的切分次数） 从-1 开始 （为了使第一次循环时 满足条件）
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <=1:
            return 0
        record =[index for index in range(-1,len(s))]
        for r in range(1,len(s)+1):
            for l in range(r):
                if s[l:r] ==s[l:r][::-1]:
                    record[r] = min(record[r],record[l]+1)
        return record[r]

# 2.交错字符串
#思路：（动态规划）
# s3 的第K (K=i+j)个位置要么是s1 的第i个位置元素 要么是s2的第j个位置元素
# 状态转移方程：dp[i][j] = (dp[i-1][j] and s3[i+j-1] == s1[i-1]) or (dp[i][j-1] and s3[i+j-1] == s2[j-1] )
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        #首先判断s1和s2的长度和是否等于s3
        if len(s1)+len(s2) != len(s3):
            return False
        dp =[[False for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        dp[0][0] = True
        for j in range(1,len(s2)+1):
            dp[0][j] =(dp[0][j-1] and  s3[j-1] == s2[j-1])
        for i in range(1,len(s1)+1):
            dp[i][0] =(dp[i-1][0] and s3[i-1] == s1[i-1])
        for i in range(1,len(s1)+1):
            for j in range(1,len(s2)+1):
                dp[i][j] = (dp[i-1][j] and s3[i+j-1] == s1[i-1]) or (dp[i][j-1] and s3[i+j-1] == s2[j-1] )
        return dp[-1][-1]
# 3.不同的子序列
#思路:(动态规划)
#状态转移方程：如果：t[i - 1] == s[j - 1]则dp[i][j] = dp[i - 1][j - 1]  + dp[i][j - 1]
# 否则：dp[i][j] = dp[i][j - 1]
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        dp = [[0 for i in range(len(s)+1)] for j in range(len(t) + 1)]
        for j in range(len(s) + 1):
            dp[0][j] = 1 #定义第一行
        for i in range(1, len(t) + 1):
            for j in range(1, len(s) + 1):
                if t[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1]
