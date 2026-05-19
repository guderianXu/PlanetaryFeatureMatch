#include "runtime/blocking_queue.h"
#include "runtime/thread_pool.h"
#include "tests/test_harness.h"

#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>
#include <thread>

static void blocking_queue_preserves_fifo_order() {
    pfm::BlockingQueue<int> queue(2);

    queue.push(1);
    queue.push(2);

    PFM_REQUIRE(queue.pop().value() == 1);
    PFM_REQUIRE(queue.pop().value() == 2);
}

static void blocking_queue_close_wakes_waiting_consumer() {
    pfm::BlockingQueue<int> queue(1);
    auto result = std::async(std::launch::async, [&queue]() {
        return queue.pop();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    queue.close();

    PFM_REQUIRE(result.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    PFM_REQUIRE(!result.get().has_value());
}

static void blocking_queue_rejects_zero_capacity() {
    PFM_REQUIRE_INVALID_ARG(pfm::BlockingQueue<int>(0));
}

static void thread_pool_runs_all_jobs() {
    pfm::ThreadPool pool(3, 8);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i) {
        pool.enqueue([&counter]() {
            counter.fetch_add(1);
        });
    }
    pool.close();
    pool.join();

    PFM_REQUIRE(counter.load() == 10);
}

static void thread_pool_rethrows_worker_exception() {
    pfm::ThreadPool pool(2, 4);

    pool.enqueue([]() {
        throw std::runtime_error("worker failed");
    });
    pool.close();

    PFM_REQUIRE_THROWS_AS(pool.join(), std::runtime_error);
}

static void thread_pool_finishes_queued_jobs_after_worker_exception() {
    pfm::ThreadPool pool(1, 4);
    std::atomic<int> counter{0};

    pool.enqueue([]() {
        throw std::runtime_error("worker failed");
    });
    for (int i = 0; i < 3; ++i) {
        pool.enqueue([&counter]() {
            counter.fetch_add(1);
        });
    }
    pool.close();

    PFM_REQUIRE_THROWS_AS(pool.join(), std::runtime_error);
    PFM_REQUIRE(counter.load() == 3);
}

static void thread_pool_rejects_zero_workers() {
    PFM_REQUIRE_THROWS_AS(pfm::ThreadPool(0, 4), std::invalid_argument);
}

void register_runtime_tests() {
    register_test("blocking queue preserves fifo order", blocking_queue_preserves_fifo_order);
    register_test("blocking queue close wakes waiting consumer", blocking_queue_close_wakes_waiting_consumer);
    register_test("blocking queue rejects zero capacity", blocking_queue_rejects_zero_capacity);
    register_test("thread pool runs all jobs", thread_pool_runs_all_jobs);
    register_test("thread pool rethrows worker exception", thread_pool_rethrows_worker_exception);
    register_test("thread pool finishes queued jobs after worker exception",
                  thread_pool_finishes_queued_jobs_after_worker_exception);
    register_test("thread pool rejects zero workers", thread_pool_rejects_zero_workers);
}
