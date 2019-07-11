# Author:Zhangbingbin 
# Time:2019/7/11

# 8.1 基本计算器
# 实现一个基本的计算器来计算一个简单的字符串表达式的值。字符串表达式仅包含非负整数，+，-，*，/ 四种运算符和空格。
# 整数除法仅保留整数部分

# 思路：遍历字符串，将一串式子看成是各种乘除法计算后式子的和
# 如：1*2-3*4---->>>>看成 1*2+（-3）*4
# 即当检测到数字（一个完整的数字，如10）时，根据这个数字前的运算符，如果为+，则直接入栈，如果为-，则-num入栈
# 如果是乘号，将栈顶弹出，top*num,再将运算结果压栈，如果是除号，将栈顶弹出，根据栈顶top的正负号再运算一波，然后将运算结果入栈
# 如果不是数字且不是空格，那么就是运算符了，op=新的运算符更新
class Solution(object):
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
if __name__ == "__main__":
    ans = Solution().calculate("141+12*1-10/1")
    print(ans)

# 8.2 整数转罗马数字
# 罗马数字包含以下七种字符：I,V,X,L,C,D,M
# 字符         数值
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000
# 例如，罗马数字2写作 Ⅱ，即为两个并列的I,12写作XII,即为X+II.27写作XXVII,即为XX+V+II
# 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写作 IIII，而是 IV。
# 数字1 在数字 5的左边，所表示的数等于大数5减小数 1 得到的数值 4 .同样的，数字 9 表示为 IX.这个特殊的规则只适用于以下六种情况：

# I可以放在V(5) 和X(10)的左边，来表示4 和9
# X可以放在L(50) 和 C(100) 的左边，来表示 40 和90
# C可以放在D(500) he M(1000)  的左边，来表示400 和900
# 给定一个罗马数字，将其转换成整数。输入确保 1 到3999 的范围内
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        lookup = {
            1:'I',
            4:'IV',
            5:'V',
            9:'IX',
            10:'X',
            40:'XL',
            50:'L',
            90:'XC',
            100:'C',
            400:'CD',
            500:'D',
            900:'CM',
            1000:'M'
        }
        res = ""
        for key in sorted(lookup.keys())[::-1]:
            a = num // key
            if a == 0:
                continue
            res += (lookup[key] * a)
            num -= a * key
            if num == 0:
                break
        return res
if __name__ == "__main__":
    res = Solution().intToRoman(1984)
    print(res)

# 8.3 有效数字
# 验证给定的字符串是否可以解释为十进制数字
# 思路：正则表达式
# 利用正则表达式匹配，整个数可分为三部分匹配，拿-13.14e-520举例：
# 1.符号（-），正则：[\+\-]?
# 2.e前面的数字（13.14），正则：(\d+\.\d+|\.\d+|\d+\.|\d+)。这里分了4种情况考虑，且匹配有先后顺序（经调试，0.0，.0，0.，0都是有效的）：
#   有小数点且小数点前后都有数字；
#   有小数点且只有小数点前面有数字；
#   有小数点且只有小数点后面有数字；
#   没有小数点（整数）。
# 3.e及其指数（e-520），正则：(e[\+\-]?\d+)? 。e0也有效。
class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        import re
        pat = re.compile(r'^[\+\-]?(\d+\.\d+|\.\d+|\d+\.|\d+)(e[\+\-]?\d+)?$')
        print(re.findall(pat, s.strip()))
        return True if len(re.findall(pat, s.strip()))  else False
if __name__ == "__main__":
    res = Solution().isNumber("17.27e98")
    print(res)