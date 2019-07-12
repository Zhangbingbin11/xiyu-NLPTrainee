# 1.基本计算器
class Solution():
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = []
        op = '+'
        i = 0
        l=len(s)
        while i < len(s):
            if s[i].isdigit():
                num = 0
                while i < len(s) and s[i].isdigit():
                    num = 10 * num + ord(s[i])
                    i += 1
                if op == '+':
                    stack.append(num)
                if op == '-':
                    stack.append(-num)
                if op == '*':
                    stack.append(num * stack.pop())
                if op == '/':
                    tmp = stack.pop()
                    if tmp < 0:
                        stack.append(-(-tmp//num))
                    else:
                        stack.append(tmp//num)
                continue
            elif s[i] != ' ':
                op = s[i]
            i += 1
        return sum(stack)

# 2.验证IP地址  (leetcode提交失败 但是pycharm运行没问题)
import re
class Solution():
    def validIPAddress(self, IP):
        if self.isIPv4(IP):
            return "IPv4"
        elif self.isIPv6(IP):
            return "IPv6"
        else:
            return "Neither"

    def isIPv4(self, IP):
        IP = IP + "."
        pv4 = re.compile(r"(([1-9]\d{0,2}|[0])\.){4}")
        m = re.fullmatch(pv4,IP)
        IP = IP[0:-1]
        if m is None:
            return False
        ip_list = list(map(lambda x: int(x), IP.split(".")))
        for part in ip_list:
            if part > 255 or part < 0:
                return False
        return True

    def isIPv6(self, IP):
        IP = IP + ":"
        pv6 = re.compile(r"(([\dA-Fa-f]{1,4}):){8}")
        m = re.fullmatch(pv6,IP)
        IP = IP[0:-1]
        if m is None:
            return False
        return True
        
# 3.最小覆盖子串
class Solution():
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import defaultdict
        lookup = defaultdict(int)
        for c in t:
            lookup[c] += 1
        start = 0
        end = 0
        min_len = float("inf")
        counter = len(t)
        res = ""
        while end < len(s):
            if lookup[s[end]] > 0:
                counter -= 1
            lookup[s[end]] -= 1
            end += 1
            while counter == 0:
                if min_len > end - start:
                    min_len = end - start
                    res = s[start:end]
                if lookup[s[start]] == 0:
                    counter += 1
                lookup[s[start]] += 1
                start += 1
        return res
        
# 4.字符串转换整数
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        return max(min(int(*re.findall(r'^[\+\-]?\d+',str.lstrip())),2**31-1),-2**31)
        
# 5.Z字形变换
class Solution():
    def convert(self, s, numRows):
        if not s:
            return ""
        if numRows == 1:
            return s
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
  
