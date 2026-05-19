#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace pfm {

/// Bounded FIFO queue that blocks producers when full and consumers when empty.
template <typename T>
class BlockingQueue {
public:
    /// Creates a queue with a positive maximum number of elements.
    ///
    /// @param capacity Maximum number of queued elements.
    /// @throws std::invalid_argument when capacity is zero.
    explicit BlockingQueue(std::size_t capacity) : _capacity(capacity) {
        if (_capacity == 0) {
            throw std::invalid_argument("blocking queue capacity must be positive");
        }
    }

    BlockingQueue(const BlockingQueue&) = delete;
    BlockingQueue& operator=(const BlockingQueue&) = delete;

    /// Pushes a value, blocking while the queue is full and open.
    ///
    /// @param value Value to append to the queue.
    /// @throws std::runtime_error when the queue is closed.
    void push(T value) {
        std::unique_lock<std::mutex> lock(_mutex);
        _not_full.wait(lock, [this]() {
            return _closed || _items.size() < _capacity;
        });
        if (_closed) {
            throw std::runtime_error("cannot push to closed blocking queue");
        }
        _items.push(std::move(value));
        _not_empty.notify_one();
    }

    /// Pops the next value, blocking while the queue is empty and open.
    ///
    /// @return Next queued value, or std::nullopt after close and drain.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(_mutex);
        _not_empty.wait(lock, [this]() {
            return _closed || !_items.empty();
        });
        if (_items.empty()) {
            return std::nullopt;
        }
        T value = std::move(_items.front());
        _items.pop();
        _not_full.notify_one();
        return value;
    }

    /// Closes the queue and wakes all waiting producers and consumers.
    void close() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _closed = true;
        }
        _not_full.notify_all();
        _not_empty.notify_all();
    }

    /// Returns the current number of queued elements.
    ///
    /// @return Thread-safe queue size snapshot.
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _items.size();
    }

private:
    std::size_t _capacity;
    mutable std::mutex _mutex;
    std::condition_variable _not_empty;
    std::condition_variable _not_full;
    std::queue<T> _items;
    bool _closed = false;
};

}  // namespace pfm
