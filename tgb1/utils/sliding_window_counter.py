from collections import Counter, deque
from typing import Any


class SlidingWindowCounter:
    """
    A class to maintain a sliding window counter 
    """
    
    def __init__(self, window_size: int):
        """
        Initialize the SlidingWindowCounter with a given window size.
        
        Args:
            window_size: The size of the sliding window.
        """
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)
        self.counter = Counter()

    def __getitem__(self, entity: Any) -> int:
        """
        Retrieve the count of a specific entity.
        
        Args:
            entity: The entity to query.
            
        Returns:
            int: The count of the queried entity.
        """
        return self.get_count(entity)

    def add(self, entity: Any) -> None:
        """
        Add a entity to the sliding window and update the counter.
        
        Args:
            entity: The entity to add.
        """
        if len(self.deque) == self.window_size:
            oldest = self.deque.popleft()
            self.counter[oldest] -= 1
            if self.counter[oldest] == 0:
                del self.counter[oldest]

        self.deque.append(entity)
        self.counter[entity] += 1

    def get_count(self, entity: Any) -> int:
        """
        Retrieve the count of a specific entity.
        
        Args:
            entity (Any): The entity to query.
            
        Returns:
            int: The count of the queried entity.
        """
        return self.counter.get(entity, 0)