# Reachability
# PATH = {〈G, s, t〉 : G is a directed graph in which there is a path from s to t}
# これはnc2か。detだもんな
# まあこれが解けるのは嬉しいよな...
# 入力に関係があたえられて...

class Reachability(task.GeneralizationTask):
  """A task with the goal of summing two numbers in binary (little-endian).

  The input is a string of the form `first_number+second_number` in
  (little-endian) binary notation (e.g., `01001+011`). The goal of the agent is
  to output the result, also in (little-endian) binary form (i.e., in the
  example `18 + 6 = 24 = 00011`). The output is padded with 0s to match the
  input length, and the end of the sum is denoted with a termination token
  (i.e., the output has values in `{0, 1, 2}`).

  Examples:
   001 + 01101    = 010112000     (4 + 22 = 26)
   1001 + 000001  = 10010120000   (9 + 32 = 41)
  """