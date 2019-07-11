# Author:Zhangbingbin 
# Time:2019/7/10
# # 7.1 无重复字符的最长子串
# # 给定一个字符串，请你找出其中不含有重复字符的最长子串的长度。
# # 思路：滑动窗口
# class Solution(object):
#     def lengthOfLongestSubstring(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         if not s:
#             return 0
#         left = 0
#         lookup = set()
#         n = len(s)
#         max_len = 0
#         cur_len = 0
#         for i in range(n):
#             cur_len += 1
#             while s[i] in lookup:
#                 lookup.remove(s[left])
#                 left += 1
#                 cur_len -= 1
#             if cur_len > max_len:
#                 max_len = cur_len
#             lookup.add(s[i])
#         return max_len
# if __name__ == "__main__":
#     ans = Solution().lengthOfLongestSubstring("absdsabcde")
#     print(ans)

# # 7.2 串联所有单词的子串
# # 给定一个字符串s 和一些长度相同的单词words 。找出s中恰好可以由 words中所有单词串联形成的子串的起始位置
# # 注意字串要与words中的单词完全匹配，中间不能有其他字符，但不需要考虑words中单词串联的顺序。
# class Solution(object):
#     def findSubstring(self, s, words):
#         """
#         :type s: str
#         :type words: List[str]
#         :rtype: List[int]
#         """
#         from collections import Counter
#         if not s or not words:
#             return []
#         one_word = len(words[0])
#         word_num = len(words)
#         n = len(s)
#         if n < one_word:return []
#         words = Counter(words)
#         res = []
#         for i in range(0, one_word):
#             cur_cnt = 0
#             left = i
#             right = i
#             cur_Counter = Counter()
#             while right + one_word <= n:
#                 w = s[right:right + one_word]
#                 right += one_word
#                 if w not in words:
#                     left = right
#                     cur_Counter.clear()
#                     cur_cnt = 0
#                 else:
#                     cur_Counter[w] += 1
#                     cur_cnt += 1
#                     while cur_Counter[w] > words[w]:
#                         left_w = s[left:left+one_word]
#                         left += one_word
#                         cur_Counter[left_w] -= 1
#                         cur_cnt -= 1
#                     if cur_cnt == word_num :
#                         res.append(left)
#         return res
# if __name__ == "__main__":
#     ans = Solution().findSubstring("barfoothefootbarman",["foo","bar"])
#     print(ans)

# # 7.3 最小覆盖子串
# # 给你一个字符串S，一个字符串T,请在字符串S 里面找出：包含T所有字母的最小子串。
# # 思路:滑动窗口
# class Solution(object):
#     def minWindow(self, s, t):
#         """
#         :type s: str
#         :type t: str
#         :rtype: str
#         """
#         from collections import defaultdict
#         lookup = defaultdict(int)
#         for c in t:
#             lookup[c] += 1
#         start = 0
#         end = 0
#         min_len = float("inf")
#         counter = len(t)
#         res = ""
#         while end < len(s):
#             if lookup[s[end]] > 0:
#                 counter -= 1
#             lookup[s[end]] -= 1
#             end += 1
#             while counter == 0:a
#                 if min_len > end - start:
#                     min_len = end - start
#                     res = s[start:end]
#                 if lookup[s[start]] == 0:
#                     counter += 1
#                 lookup[s[start]] += 1
#                 start += 1
#         return res
# if __name__ == "__main__":
#     ans = Solution().minWindow("ABAACBAB","ABC")
#     print(ans)

# # 7.4Pandas基础
# # 1.如何将numpy数组转换为给定形状的dataframe
# # 2.从dataframe中获取c列最大值所在的行号
# import pandas as pd
# import numpy as np
# ser = pd.Series(np.random.randint(1,10,35))
# df = pd.DataFrame(ser.values.reshape(7,5))
# print(df)
#
# df = pd.DataFrame({
#     'a':range(5),
#     'b':[6,5,4,3,2],
#     'c':np.random.randint(1,10,5)
# })
# print(df.head())
# print(df.loc[df.a==np.max(df.a)]) #从dataframe中找到a列最大值对应的行
# print(df.where(df.c==np.max(df.c))) # 从dataframe中获取c列最大值所在的行号