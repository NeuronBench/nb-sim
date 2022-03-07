use std::fmt::Debug;

/// A fixed-size buffer that can be written to as many times as you like,
/// dropping the least-recently-added element when capacity is reached.
#[derive(Clone, Debug)]
pub struct RingBuffer<T> {
    pub inner: Vec<T>,
    pub write_cursor: usize,
    pub len: usize,
}

impl<T: Clone + Debug> RingBuffer<T> {
    pub fn new(size: usize, default: T) -> RingBuffer<T> {
        let mut inner = Vec::with_capacity(size);
        for _ in 0..size {
            inner.push(default.clone());
        }
        RingBuffer {
            inner,
            write_cursor: 0,
            len: 0,
        }
    }

    pub fn write(&mut self, value: T) {
        let ind = self.write_cursor;
        self.inner[ind] = value;
        self.write_cursor += 1;
        if self.len < self.inner.len() {
            self.len += 1;
        }
        if self.write_cursor == self.inner.len() {
            self.write_cursor = 0;
        }
    }

    pub fn contents(&self) -> Vec<T> {
        if self.len() == self.inner.len() {
            let oldest_elements = &self.inner[self.write_cursor..self.inner.len()];
            let newest_elements = &self.inner[0..self.write_cursor];
            oldest_elements
                .iter()
                .chain(newest_elements.iter())
                .map(|e| e.clone())
                .collect()
        } else {
            self.inner[0..self.len()]
                .into_iter()
                .map(|e| e.clone())
                .collect()
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;
    #[test]
    fn test_size() {
        let b = RingBuffer::new(5, 0);
        assert_eq!(b.len(), 0);
        assert_eq!(b.contents().len(), 0);
    }

    #[test]
    fn test_writes() {
        let mut b = RingBuffer::new(5, 10);
        assert_eq!(b.contents(), vec![]);

        b.write(1);
        b.write(2);
        b.write(3);
        assert_eq!(b.inner, vec![1, 2, 3, 10, 10]);
        assert_eq!(b.contents(), vec![1, 2, 3]);

        b.write(4);
        b.write(5);
        b.write(6);
        assert_eq!(b.contents(), vec![2, 3, 4, 5, 6]);
    }
}
