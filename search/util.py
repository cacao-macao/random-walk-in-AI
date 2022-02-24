from collections import deque
import heapq
import fibheap


class LimitedDict:
    """ A dictionary with limited size.
    If the number of items in the dictionary reaches the limit, then new items will not
    be added.
    """
    def __init__(self, size):
        """ Initialize a new limited dict instance.

        @param size(int): Limiting size for the dictionary.
        """
        self._size = size
        self._dict = {}

    def __setitem__(self, key, value):
        """ Maybe insert the key:value pair in the dictionary.
        If the key is already in the dictionary, then modify the value.
        If the number of items in the dictionary has not reached the limit, then add
        the new key:value pair.
        Otherwise drop the new key:value pair.
        """
        if key in self._dict:
            self._dict[key] = value
        if len(self._dict) < self._size:
            self._dict[key] = value
    
    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict


class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self._data = deque()

    def push(self, item):
        "Push 'item' onto the stack"
        self._data.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self._data.pop()

    def peek(self):
        "Peek the most recently pushed item."
        return self._data[-1]

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self._data) == 0
    
    def __len__(self):
        return len(self._data)


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self._data = deque()

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self._data.appendleft(item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self._data.pop()

    def peek(self):
        "Peek the earliest pushed item."
        return self._data[0]

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self._heap = []
        self._count = 0

    def push(self, item, priority):
        entry = (priority, self._count, item)
        heapq.heappush(self._heap, entry)
        self._count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self._heap)
        return item

    def peek(self):
        "Peek the item at the top of the heap."
        return self._heap[0][-1]

    def isEmpty(self):
        return len(self._heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self._heap):
            if i == item:
                # if p <= priority:
                #     break
                del self._heap[index]
                self._heap.append((priority, c, item))
                heapq.heapify(self._heap)
                break
        else:
            self.push(item, priority)

    def __len__(self):
        return len(self._heap)

    # def __contains__(self, item):
    #     return item in (foo[-1] for foo in self._heap)


class FibPriorityQueue:
    def  __init__(self):
        self._fheap = fibheap.Fheap()
        self._itemToNode = {}
        self._count = 0

    def push(self, item, priority):
        entry = (priority, self._count, item)
        node = fibheap.Node(entry)
        self._fheap.insert(node)
        self._itemToNode[item] = node
        self._count += 1

    def pop(self):
        node = self._fheap.extract_min()
        _, _, item = node.key
        self._itemToNode.pop(item)
        return item

    def isEmpty(self):
        return self._fheap.num_nodes == 0

    def update(self, item, priority):
        node = self._itemToNode[item]
        if priority < node.key[0]:
            self._fheap.decrease_key(node, (priority, node.key[1], node.key[2]))

    def __len__(self):
        return self._fheap.num_nodes


class PriorityQueueForSMA:
    def __init__(self, nodes={}):
        heap = []
        reverse_index = {}
        i = 0
        for node, value in nodes.items():
            heap.append((value, node))
            reverse_index[node] = i
            i += 1
        self._heap = heap
        self._reverse_index = reverse_index
        self._count = 0
        self._heapify()
        self._verify()

    def push(self, node, score):
        entry = (score, self._count, node)
        self._heap.append(entry)
        self._count += 1
        j = len(self._heap) - 1
        self._reverse_index[node] = j
        self._bubble(j)

    def pop(self):
        """ Remove and return the current best node. """
        H, R = self._heap, self._reverse_index
        # Elements in H are (score, count, node) tuples
        _, _, result = H[0]
        if len(H) == 1:
            R.clear()
            H.clear()
        else:
            R.pop(H[0][-1])
            H[0] = H[-1]
            R[H[0][-1]] = 0
            H.pop()
            self._sink(0, verify=False)
        return result

    def peek(self):
        "Peek the item at the top of the heap."
        return self._heap[0]

    def update(self, node, score):
        """
        Updates the value of `node` with `score`, maintaining the heap
        and reverse map invariants.
        """
        i = self._reverse_index[node]
        oldval = self._heap[i][0]
        self._heap[i] = (score, self._heap[i][1], self._heap[i][2])
        if score < oldval:
            self._bubble(i, verify=False)
        elif score > oldval:
            self._sink(i, verify=False)
        # self._verify()

    def remove(self, node):
        if node not in self._reverse_index:
            return
        i = self._reverse_index[node]
        oldval = self._heap[i][0]
        score = self._heap[-1][0]
        H, R = self._heap, self._reverse_index
        if len(H) == 1:
            R.clear()
            H.clear()
        elif i == len(H) - 1:
            R.pop(H[i][-1])
            H.pop()
        else:
            R.pop(H[i][-1])
            H[i] = H[-1]
            R[H[i][-1]] = i
            H.pop()
            if score < oldval:
                self._bubble(i, verify=False)
            elif score > oldval:
                self._sink(i, verify=False)

    def _sink(self, i, verify=False):
        """
        Moves the item at index `i` downward, maintaining the
        heap invariant and the index map
        """
        # Elements in H are (score, count, node) tuples
        H = self._heap
        N = len(H)
        while i < N:
            li, ri = (2 * (i + 1) - 1), (2 * (i + 1))
            min_score = H[i][0]
            j = i
            if li < N and H[li][0] < min_score:
                j = li
                min_score = H[li][0]
            if ri < N and H[ri][0] < min_score:
                j = ri
            if j != i:
                index = self._reverse_index
                # Update the index map
                index[H[j][-1]], index[H[i][-1]] = i, j
                # Do the swap
                H[j], H[i] = H[i], H[j]
                # Next iteration
                i = j
            else:
                break
        if verify:
            self._verify()

    def _bubble(self, i, verify=False):
        """
        Moves the item at index `i` upward, maintaining the
        heap invariant and the index map
        """
        # Elements in H are (score, count, node) tuples
        H = self._heap
        while i > 0:
            p = (i - 1) // 2
            if H[p][0] > H[i][0]:
                # Update the reverse index
                index = self._reverse_index
                index[H[p][-1]], index[H[i][-1]] = i, p
                # Do the swap
                H[i], H[p] = H[p], H[i]
                # Next iteration
                i = p
            else:
                break
        if verify:
            self._verify()

    def _heapify(self):
        """ Builds a max heap in `self._heap` """
        H = self._heap
        j = len(H) // 2
        sink = self._sink
        for i in range(j, -1, -1):
            sink(i)

    def _verify(self):
        """ Verifies the heap and reverse index invariants """
        H = self._heap
        N = len(H)
        R = self._reverse_index
        for i in range(N):
            assert(R[H[i][-1]] == i)
            li, ri = (2 * (i + 1) - 1), (2 * (i + 1))
            if li < N:
                assert(H[i][0] <= H[li][0])
            if ri < N:
                assert(H[i][0] <= H[ri][0])

    def __repr__(self):
        return 'Frontier({})'.format(repr(self._heap))

    def isEmpty(self):
        return len(self._heap) == 0

    def __len__(self):
        return len(self._heap)

    def __contains__(self, item):
        return item in self._reverse_index

#