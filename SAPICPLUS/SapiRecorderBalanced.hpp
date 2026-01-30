/**
 * SapiAgent Three-Dot Flow Recorder - BALANCED VERSION (C++)
 * Version: 8.1.0 - ENHANCED MULTI-THREADED ARCHITECTURE
 *
 * Optimized for Intel i5-10600K (6 cores / 12 threads)
 *
 * Thread Architecture (6 threads):
 * - Thread 1: Main Thread - SDL2 rendering & window management
 * - Thread 2: Event Processing Thread - Decoupled event handling
 * - Thread 3: Sampling Thread - 125Hz mouse position polling
 * - Thread 4: Computation Thread - Segment metadata calculations
 * - Thread 5-6: File I/O Thread Pool - Parallel JSON saving (2 workers)
 *
 * Dependencies: SDL2, SDL2_ttf
 */

#ifndef SAPI_RECORDER_BALANCED_HPP
#define SAPI_RECORDER_BALANCED_HPP

#include <string>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <functional>
#include <shared_mutex>

// =============================================================================
// CONFIGURATION CONSTANTS
// =============================================================================

namespace Config {
    // Visual
    constexpr int DOT_RADIUS = 30;
    constexpr uint32_t DOT_A_COLOR = 0x22c55eFF;  // Green - origin
    constexpr uint32_t DOT_B_COLOR = 0x3b82f6FF;  // Blue - current target
    constexpr uint32_t DOT_C_COLOR = 0xf97316FF;  // Orange - next target
    constexpr uint32_t BACKGROUND_COLOR = 0x000000FF;  // Black
    constexpr uint32_t LINE_COLOR = 0x333333FF;
    constexpr uint32_t TEXT_COLOR = 0xFFFFFFFF;

    // Screen constraints
    constexpr int SCREEN_WIDTH = 2560;
    constexpr int SCREEN_HEIGHT = 1400;
    constexpr int SCREEN_MARGIN = 10;

    // Session
    constexpr int TARGET_SEGMENTS = 192;
    constexpr int SESSION_DURATION_MS = 1800000;  // 30 minute max

    // Sampling
    constexpr int SAMPLE_RATE_HZ = 125;
    constexpr double SAMPLE_INTERVAL_MS = 1000.0 / SAMPLE_RATE_HZ;

    // Output directory (configurable)
    inline std::string OUTPUT_DIR = "D:/V12/V12_Anchors_Continuous/three_dot_flow";

    // Minimum trigger interval to prevent double-triggering
    constexpr int MIN_TRIGGER_INTERVAL_MS = 100;

    // Path count for random selection
    constexpr int PATH_COUNT = 15;

    // Enhanced Threading Configuration (optimized for i5-10600K: 6 cores / 12 threads)
    constexpr int FILE_IO_WORKER_COUNT = 2;      // Parallel file writers
    constexpr int COMPUTATION_QUEUE_SIZE = 32;   // Max pending computations
    constexpr int EVENT_QUEUE_SIZE = 64;         // Max pending events

    // Thread priorities (platform-specific, hint only)
    constexpr int PRIORITY_SAMPLING = 2;    // Highest - timing critical
    constexpr int PRIORITY_EVENT = 1;       // High - responsiveness
    constexpr int PRIORITY_COMPUTE = 0;     // Normal
    constexpr int PRIORITY_FILE_IO = -1;    // Lower - background work
}

// =============================================================================
// DISTANCE THRESHOLDS AND NAMES
// =============================================================================

namespace Distance {
    constexpr std::array<int, 12> THRESHOLDS = {27, 57, 117, 177, 237, 297, 357, 417, 477, 537, 597, 657};

    inline const std::array<std::string, 12> NAMES = {
        "XXXS", "XXS", "XS", "S", "XXXM", "XXM", "XM", "M", "XXXL", "XXL", "XL", "L"
    };
}

// =============================================================================
// ORIENTATIONS
// =============================================================================

namespace Orientation {
    inline const std::array<std::string, 8> NAMES = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};

    // Screen angle ranges for each orientation
    struct AngleRange {
        double lo;
        double hi;
    };

    inline const std::map<std::string, AngleRange> SCREEN_ANGLE_RANGES = {
        {"E",  {-22.5, 22.5}},
        {"SE", {22.5, 67.5}},
        {"S",  {67.5, 112.5}},
        {"SW", {112.5, 157.5}},
        {"W",  {157.5, 180.0}},  // Also covers -180 to -157.5
        {"NW", {-157.5, -112.5}},
        {"N",  {-112.5, -67.5}},
        {"NE", {-67.5, -22.5}},
    };

    // Unit vectors for each orientation
    struct Vector2D {
        double x;
        double y;
    };

    inline const std::map<std::string, Vector2D> VECTORS = {
        {"N",  {0.0, -1.0}},
        {"NE", {0.707, -0.707}},
        {"E",  {1.0, 0.0}},
        {"SE", {0.707, 0.707}},
        {"S",  {0.0, 1.0}},
        {"SW", {-0.707, 0.707}},
        {"W",  {-1.0, 0.0}},
        {"NW", {-0.707, -0.707}},
    };
}

// =============================================================================
// TURN CATEGORIES
// =============================================================================

namespace Turn {
    struct AngleRange {
        double lo;
        double hi;
    };

    inline const std::map<std::string, AngleRange> CATEGORIES = {
        {"straight",      {-25.7, 25.7}},
        {"slight_right",  {25.7, 77.1}},
        {"hard_right",    {77.1, 128.6}},
        {"reverse_right", {128.6, 180.0}},
        {"slight_left",   {-77.1, -25.7}},
        {"hard_left",     {-128.6, -77.1}},
        {"reverse_left",  {-180.0, -128.6}},
    };
}

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Path segment from precomputed data
struct PathSegment {
    int distance;
    std::string orientation;
    int start_x;
    int start_y;
};

// Dot position and metadata
struct Dot {
    int x;
    int y;
    int distance;
    std::string orientation;
    int distance_group;
    std::string distance_name;

    Dot() : x(0), y(0), distance(0), distance_group(0) {}
};

// A single trajectory point
struct TrajectoryPoint {
    int64_t timestamp;
    int x;
    int y;
};

// Segment metadata for saving
struct SegmentData {
    int segment_id;
    int64_t timestamp_start;
    int64_t timestamp_end;
    int64_t duration_ms;

    Dot dot_A;
    Dot dot_B;
    Dot dot_C;

    // AB segment info
    double ab_distance;
    int ab_distance_group;
    std::string ab_distance_name;
    double ab_angle_deg;
    std::string ab_orientation;
    int ab_orientation_id;

    // BC segment info
    double bc_distance;
    int bc_distance_group;
    std::string bc_distance_name;
    double bc_angle_deg;
    std::string bc_orientation;
    int bc_orientation_id;

    // Turn info
    double turn_angle_deg;
    std::string turn_category;

    // Trajectory
    std::vector<TrajectoryPoint> trajectory;

    // Processing state
    bool computed = false;
};

// Raw segment data (before computation) - passed from main to computation thread
struct RawSegmentData {
    int segment_id;
    int64_t timestamp_start;
    int64_t timestamp_end;

    Dot dot_A;
    Dot dot_B;
    Dot dot_C;

    std::vector<TrajectoryPoint> trajectory;
};

// Thread statistics for monitoring
struct ThreadStats {
    std::atomic<uint64_t> samples_processed{0};
    std::atomic<uint64_t> events_processed{0};
    std::atomic<uint64_t> segments_computed{0};
    std::atomic<uint64_t> files_saved{0};
    std::atomic<uint64_t> queue_overflows{0};

    // Timing stats (in microseconds)
    std::atomic<uint64_t> avg_sample_latency_us{0};
    std::atomic<uint64_t> avg_compute_time_us{0};
    std::atomic<uint64_t> avg_save_time_us{0};
};

// =============================================================================
// THREAD-SAFE QUEUE
// =============================================================================

template<typename T>
class ThreadSafeQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms < 0) {
            cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        } else {
            if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [this] { return !queue_.empty() || shutdown_; })) {
                return false;
            }
        }

        if (shutdown_ && queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }

    bool isShutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool shutdown_ = false;
};

// =============================================================================
// THREAD POOL (For parallel file I/O)
// =============================================================================

template<typename Task>
class ThreadPool {
public:
    explicit ThreadPool(size_t num_workers, const std::string& name = "Worker")
        : name_(name), running_(false) {
        workers_.reserve(num_workers);
    }

    ~ThreadPool() {
        stop();
    }

    void start() {
        running_ = true;
        for (size_t i = 0; i < workers_.capacity(); ++i) {
            workers_.emplace_back(&ThreadPool::workerLoop, this, i);
        }
        std::cout << "[ThreadPool:" << name_ << "] Started " << workers_.size() << " workers\n";
    }

    void stop() {
        running_ = false;
        task_queue_.shutdown();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        std::cout << "[ThreadPool:" << name_ << "] Stopped\n";
    }

    void submit(Task task) {
        if (running_) {
            task_queue_.push(std::move(task));
        }
    }

    size_t pendingTasks() const {
        return task_queue_.size();
    }

    size_t workerCount() const {
        return workers_.size();
    }

private:
    void workerLoop(size_t worker_id) {
        while (running_) {
            Task task;
            if (task_queue_.pop(task, 100)) {
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "[ThreadPool:" << name_ << ":Worker" << worker_id
                              << "] Exception: " << e.what() << "\n";
                }
            }
        }

        // Drain remaining tasks
        Task task;
        while (task_queue_.pop(task, 0)) {
            try {
                task();
            } catch (...) {}
        }
    }

    std::string name_;
    std::atomic<bool> running_;
    std::vector<std::thread> workers_;
    ThreadSafeQueue<Task> task_queue_;
};

// =============================================================================
// LOCK-FREE SPSC RING BUFFER (For high-frequency sampling data)
// =============================================================================

template<typename T, size_t Capacity>
class SPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

public:
    SPSCRingBuffer() : head_(0), tail_(0) {
        buffer_.resize(Capacity);
    }

    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) & (Capacity - 1);

        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Full
        }

        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // Empty
        }

        item = buffer_[tail];
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail + Capacity) & (Capacity - 1);
    }

private:
    std::vector<T> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

namespace Helpers {
    // Convert degrees to radians
    inline double deg2rad(double deg) {
        return deg * M_PI / 180.0;
    }

    // Convert radians to degrees
    inline double rad2deg(double rad) {
        return rad * 180.0 / M_PI;
    }

    // Compute signed angle from v1 to v2 in degrees
    inline double angleBetweenVectors(double v1x, double v1y, double v2x, double v2y) {
        double angle1 = std::atan2(v1y, v1x);
        double angle2 = std::atan2(v2y, v2x);
        double diff = angle2 - angle1;

        while (diff > M_PI) diff -= 2.0 * M_PI;
        while (diff < -M_PI) diff += 2.0 * M_PI;

        return rad2deg(diff);
    }

    // Convert angle in degrees to orientation string
    inline std::string getOrientationFromAngle(double angle_deg) {
        if (angle_deg > 157.5 || angle_deg <= -157.5) {
            return "W";
        }

        for (const auto& [orient, range] : Orientation::SCREEN_ANGLE_RANGES) {
            if (orient == "W") continue;
            if (angle_deg >= range.lo && angle_deg < range.hi) {
                return orient;
            }
        }
        return "E";
    }

    // Convert orientation string to ID (0-7)
    inline int getOrientationId(const std::string& orient) {
        for (size_t i = 0; i < Orientation::NAMES.size(); ++i) {
            if (Orientation::NAMES[i] == orient) {
                return static_cast<int>(i);
            }
        }
        return 0;
    }

    // Get distance group ID and name
    inline std::pair<int, std::string> getDistanceGroup(double distance) {
        const auto& thresholds = Distance::THRESHOLDS;
        const auto& names = Distance::NAMES;

        for (size_t i = 0; i < thresholds.size(); ++i) {
            if (i == 0) {
                double mid = (thresholds[0] + thresholds[1]) / 2.0;
                if (distance < mid) {
                    return {static_cast<int>(i), names[i]};
                }
            } else if (i == thresholds.size() - 1) {
                return {static_cast<int>(i), names[i]};
            } else {
                double mid_low = (thresholds[i-1] + thresholds[i]) / 2.0;
                double mid_high = (thresholds[i] + thresholds[i+1]) / 2.0;
                if (distance >= mid_low && distance < mid_high) {
                    return {static_cast<int>(i), names[i]};
                }
            }
        }

        return {static_cast<int>(thresholds.size() - 1), names.back()};
    }

    // Get turn category from angle
    inline std::string getTurnCategory(double turn_angle) {
        for (const auto& [category, range] : Turn::CATEGORIES) {
            if (turn_angle >= range.lo && turn_angle < range.hi) {
                return category;
            }
        }
        return "straight";
    }

    // Calculate distance between two points
    inline double distance(int x1, int y1, int x2, int y2) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        return std::sqrt(dx * dx + dy * dy);
    }

    // Check if point is inside a circle
    inline bool isInsideCircle(int px, int py, int cx, int cy, int radius) {
        int dx = px - cx;
        int dy = py - cy;
        return (dx * dx + dy * dy) <= (radius * radius);
    }

    // Get current timestamp in milliseconds
    inline int64_t getCurrentTimeMs() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }

    // Generate session ID string
    inline std::string generateSessionId() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time_t);

        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "session_%Y_%m_%d_%H%M%S", &tm);
        return std::string(buffer);
    }
}

// =============================================================================
// EVENT TYPES FOR INTER-THREAD COMMUNICATION
// =============================================================================

enum class RecorderEvent {
    None,
    StartRecording,
    CompleteSegment,
    EndSession,
    UpdateUI,
    DotShifted,
    SegmentSaved,
    Error
};

struct MouseEvent {
    int64_t timestamp;
    int x;
    int y;
    RecorderEvent event_type;
};

// High-frequency sample data (lock-free transfer)
struct SampleData {
    int64_t timestamp;
    int x;
    int y;
    bool valid;
};

// Callback types for thread communication
using MouseSampleCallback = std::function<void(int64_t, int, int)>;
using EntryEventCallback = std::function<void(RecorderEvent, int64_t, int, int)>;
using SegmentCompleteCallback = std::function<void(const SegmentData&)>;
using ComputationCompleteCallback = std::function<void(SegmentData)>;

// =============================================================================
// CPU AFFINITY HELPER (Platform-specific optimization)
// =============================================================================

namespace ThreadUtils {
    // Set thread name (for debugging)
    inline void setThreadName([[maybe_unused]] const std::string& name) {
#ifdef __linux__
        pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#elif defined(__APPLE__)
        pthread_setname_np(name.c_str());
#endif
    }

    // Get number of hardware threads
    inline unsigned int getHardwareThreads() {
        unsigned int threads = std::thread::hardware_concurrency();
        return threads > 0 ? threads : 4;  // Default to 4 if detection fails
    }

    // High-resolution sleep (more precise than std::this_thread::sleep_for)
    inline void preciseSleep(std::chrono::nanoseconds duration) {
        auto start = std::chrono::high_resolution_clock::now();
        auto end = start + duration;

        // Busy-wait for short sleeps, yield for longer ones
        while (std::chrono::high_resolution_clock::now() < end) {
            if ((end - std::chrono::high_resolution_clock::now()) > std::chrono::microseconds(100)) {
                std::this_thread::yield();
            }
        }
    }
}

#endif // SAPI_RECORDER_BALANCED_HPP
