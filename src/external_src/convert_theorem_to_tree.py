def parse_as_list(term):
  term = term.strip()
  stack = [[]]
  assert term[0] == '(', ('Syntax error at %d' % starting_index)
  for char in term:
    if char == '(':
      stack.append([])
      continue
    elif char == ')':
      # print stack
      item = stack.pop()
      if stack[-1] and stack[-1][-1] == '':
        stack[-1][-1] = item
      else:
        stack[-1].append(item)
    elif char == ' ':
      if stack[-1] and stack[-1][-1] != '':
        stack[-1].append('')
    else:
      if not stack[-1] or isinstance(stack[-1][-1], list):
        stack[-1].append(char)
      else:
        stack[-1][-1] += char
  assert len(stack) == 1
  assert len(stack[0]) == 1
  return stack[0][0]
