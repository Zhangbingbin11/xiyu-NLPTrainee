#1.1有效的括号
#思路：依次将这些字符入栈，判断栈顶元素与当前元素是否是对应上的，是则pop,不是则入栈，到最后判断栈是否为空
def isValid(s):
    """
    :type s: str
    :rtype: bool
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for i in s:
        if i in mapping and mapping[i] == stack[-1]:
            stack.pop()
        else:
            stack.append(i)
    return  len(stack) == 0
ans=isValid("(()()")
print(ans)

#1.2括号生成（回溯法）
"""思路：每次增加括号时需要判断之前字符串中左右括号的个数
判断增加“(”还是“)”的依据：
比较之前字符串中左括号数小于N，则增加左括号；若之前字符串中右括号小于左括号，则增加右括号"""
class Solution(object):
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            if len(S) == 2 * N:
                ans.append(S)
                return
            if left < N:
                backtrack(S+'(', left+1, right)
            if right < left:
                backtrack(S+')', left, right+1)
        backtrack()
        return ans
a = Solution()
ans = a.generateParenthesis(3)
print(ans)

# 1.3最长的有效括号
"""思路1:使用栈进行操作，如果是左括号，直接入stack，如果右括号，如果stack里没有元素匹对，
说明有效括号已经结束，更新起始位置，有元素匹对pop出一个左括号匹对，如果此时没了，不能保
证不继续有效括号，所以根据当前的最长距离去更新maxlen，如果此时还有 则计算与栈顶的索引
相减来计算长度。"""
class Solution():
    def longestValidParentheses(self, s):
        ans = 0
        n = len(s)
        stack = []
        st = 0
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            else:
                if len(stack) == 0:
                    st = i + 1
                    continue
                else:
                    stack.pop()
                    if len(stack) == 0:
                        ans = max(ans, i - st + 1)
                    else:
                        ans = max(ans, i - stack[-1])
        return ans
"""思路2：动态规划，dp[i]表示从开始到 i 点的最长有效括号的长度,我们每两个字符检查一次，"""
class Solution():
    def longestValidParentheses(self, s: str):
        n = len(s)
        if n == 0:
            return 0
        dp = [0] * n
        res = 0
        #每两个字符检查一次
        for i in range(n):
            if i>0 and s[i] == ")":
                #第一种情况：i-1处与i处匹配
                if  s[i - 1] == "(":
                    dp[i] = dp[i - 2] + 2
                 #第二种情况：不匹配   且  除去中间有效括号 前面有左括号
                elif s[i - 1] == ")" and i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == "(":
                    dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2] #则dp[i] = 这一段的dp + 那个左括号前面的 dp
                if dp[i] > res:
                    res = dp[i] #取大值
        return res
a = Solution()
res = a.longestValidParentheses(")((()))()")
print(res)

#1.4替换空格
# 方法一：直接用接口
class Solution():
    def replaceSpace(self,s):
        return s.replace(' ','%20')
a = Solution()
str =a.replaceSpace("We Are Happy.")
print(str)
#方法二：在python中字符串不可变 时间复杂度O(n),空间O(n)
class Solution():
    def replaceSpace(self,s):
        res = ''
        for i in s:
            if i == ' ':
                res += '%20'
            else:
                res += i
        return res
a = Solution()
res = a.replaceSpace("WE ARE HAPPY.")
print(res)
#方法三：通用(两个指针p1,p2) 时间O(n^2)
def replaceSpace(s):
    p1 = len(s) - 1
    res = list(s)
    n = s.count(' ')
    res += [0] * n * 2
    p2 = len(res) - 1
    while p1 != p2:
        if res[p1] == ' ':
            res[p2] = '0'
            res[p2 - 1] = '2'
            res[p2 - 2] = '%'
            p2 -= 3
        else:
            res[p2] = res[p1]
            p2 -= 1
        p1 -= 1
    return ''.join(res)
s_res = replaceSpace("we are happy.")
print(s_res)

#1.5Pandas基础
pandas 遍历有以下三种方法
.iterrows（）：在单独的变量中返回索引和行项目，但显着较慢
.itertuples（）：快于.iterrows（），但将索引与行项目一起返回，ir [0]是索引
zip：最快，但不能访问该行的索引
import numpy as np
import pandas as pd
# pd.show_versions()
mylist = list('abcdefghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist,myarr))
# print(mydict)
#列表，数组，字典转化为series
ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series(mydict)
s1 = ser1[:10]
s2 = ser2[10:]
print(pd.concat([s1,s2],axis=0))#头尾拼接两个series
sA = pd.Series([1,2,3,4,5,6])
sB = pd.Series([5,6,7,8,9,10])
print(sA[~sA.isin(sB)])#元素在sA不在sB
