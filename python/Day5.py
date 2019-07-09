# Author:Zhangbingbin 
# Time:2019/7/8
# 5.1复原IP地址
#给定一个只包含数字的字符串，复原它并返回所有可能的IP地址格式
#思路：我们知道IP地址分为4个小段，每一段都是再0~255这个区间内（注意：第一小段不能为0）
#有个细节：对于前三段做一样的处理（结尾加“.”）,而最后一段单独考虑，所以就成的边界条件之一。

#方法一：（递归）
class Solution:
    def _restoreIpAddresses(self, s, n, index, ip, result):
        if n == 0:
            if index == len(s):
                result.append(ip)
            return
        def isNum(num):
            if 0 <= int(num) <= 255 and str(int(num)) == num:
                return True
            return False
        for i in range(index + 1, len(s) + 1):
            if isNum(s[index:i]):
                self._restoreIpAddresses(s, n - 1, i, s[index:i] if ip == "" else ip + '.' + s[index:i], result)
            else:
                break

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        result = list()
        if not s and 4 > len(s) > 12:
            return result

        self._restoreIpAddresses(s, 4, 0, "", result)
        return result

#方法二：（迭代）
class Solution:
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def isNum(num):
            if num and 0 <= int(num) <= 255 and str(int(num)) == num:
                return True
            return False
        result = list()
        for i in range(1, 4):
            w1 = s[:i]
            if not isNum(w1):
                continue
            for j in range(i + 1, i + 4):
                w2 = s[i:j]
                if not isNum(w2):
                    continue
                for k in range(j + 1, j + 4):
                    w3, w4 = s[j:k], s[k:]
                    print(w3, w4)
                    if not isNum(w3) or not isNum(w4):
                        continue
                    result.append(w1 + '.' + w2 + '.' + w3 + '.' + w4)
        return result

# 5.2 验证IP地址
# 编写一个函数来验证输入的字符串是否是有效的 IPv4 或 IPv6 地址。
# IPv4 地址由十进制数和点来表示，每个地址包含4个十进制数，其范围为 0 - 255， 用(".")分割。比如，172.16.254.1；同时，IPv4 地址内
# 的数不会以 0 开头。比如，地址 172.16.254.01 是不合法的
#IPv6 地址由8组16进制的数字来表示，每组表示 16 比特。这些组数字通过 (":")分割。比如2001:0db8:85a3:0000:0000:8a2e:0370:7334
# 是一个有效的地址。而且，我们可以加入一些以 0 开头的数字，字母可以使用大写，也可以是小写。所以2001:db8:85a3:0:0:8A2E:0370:7334
# 也是一个有效的 IPv6 address地址 (即，忽略 0 开头，忽略大小写)。
#然而，我们不能因为某个组的值为 0，而使用一个空的组，以至于出现 (::) 的情况。 比如， 2001:0db8:85a3::8A2E:0370:7334
# 是无效的 IPv6 地址。同时，在 IPv6 地址中，多余的 0 也是不被允许的。比如02001:0db8:85a3:0000:0000:8a2e:0370:7334是无效的。

# 思路：IPv4先用正则匹配出格式，再split一下每一段确保在0-255之间 IPv6直接用正则就好 小技巧：可以先给IP补上"."号或":"号，以方便匹配
class Solution:
    def validIPAddress(self, IP: str) -> str:
        if self.isIPv4(IP):
            return "IPv4"
        elif self.isIPv6(IP):
            return "IPv6"
        else:
            return "Neither"

    def isIPv4(self, IP: str) -> bool:
        import re
        # 格式校验
        IP = IP + "."
        pv4 = re.compile(r"(([1-9]\d{0,2}|[0])\.){4}") # # .号要记得转义\
        m = pv4.fullmatch(IP)  # 全匹配
        IP = IP[0:-1]  # 去掉尾部.号
        if m is None:
            return False
        # 校验大小
        # print(m.group())
        ip_list = list(map(lambda x: int(x), IP.split(".")))
        for part in ip_list:
            if part > 255 or part < 0:
                return False
        return True

    def isIPv6(self, IP: str) -> bool:
        import re
        IP = IP + ":"
        pv6 = re.compile(r"(([\dA-Fa-f]{1,4}):){8}")
        m = pv6.fullmatch(IP)
        IP = IP[0:-1]
        if m is None:
            return False
        return True
a = Solution().validIPAddress("212:21:2:3:9:8:7:5")
print(a)


# 5.3 中文处理之年份转换
# -*- coding: cp936 -*-
import re
m0 =  "在一九四九年新中国成立"
m1 =  "比一九九零年低百分之五点二"
m2 = '人一九九六年击败俄军,取得实质独立'

def fuc(m):
    a = re.findall(u"[\u96f6|\u4e00|\u4e8c|\u4e09|\u56db|\u4e94|\u516d|\u4e03|\u516b|\u4e5d]+\u5e74", m)
    if a:
        for key in a:
            return(key)
    else:
        return("NULL")
mo1 = fuc(m0)
print(mo1)
# fuc(m1)
# fuc(m2)

numHash = {}
numHash['零'] = '0'
numHash['一'] = '1'
numHash['二'] = '2'
numHash['三'] = '3'
numHash['四'] = '4'
numHash['五'] = '5'
numHash['六'] = '6'
numHash['七'] = '7'
numHash['八'] = '8'
numHash['九'] = '9'

def change2num(words):
    newword = ''
    for key in words:
        if key in numHash:
            newword += numHash[key]
        else:
            newword += key
    return newword
m0_num = change2num(fuc(m0))
print(m0_num)


# 5.4 Pandas基础
#1.从Series中找出包含两个以上的元音字母的单词
#2.如何过滤series 中的有效电子邮件
import pandas as pd
ser = pd.Series(['Apple','Orange','Plan','Python','Money'])
def count(x):
    aims = 'aeiou'
    c = 0
    for i in x:
        if i in aims:
            c +=1
    return c
counts = ser.str.lower().map(lambda x:count(x)) #先要将ser序列中的字母变小写（.str.lower()）
print(ser[counts>=2])  #从Series中找出包含两个以上的元音字母的单词

#如何过滤series 中的有效电子邮件
emails = pd.Series(['buying books at amazom.com',
                    'rameses@egypt.com',
                    'matt@t.co',
                    'narendra@modi.com'])
import re
pattern = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
valid = emails.str.findall(pattern,flags=re.IGNORECASE)
print([x[0] for x in valid if len(x)])
