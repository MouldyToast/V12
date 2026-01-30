/**
 * SapiAgent Three-Dot Flow Recorder - BALANCED VERSION (C++)
 * Version: 8.1.0 - ENHANCED MULTI-THREADED ARCHITECTURE
 *
 * Optimized for Intel i5-10600K (6 cores / 12 threads)
 *
 * Thread Architecture:
 * - Thread 1: Main Thread - SDL2 rendering & window management
 * - Thread 2: Event Processing Thread - Decoupled event handling
 * - Thread 3: Sampling Thread - 125Hz mouse position polling
 * - Thread 4: Computation Thread - Segment metadata calculations
 * - Thread 5-6: File I/O Thread Pool - Parallel JSON saving (2 workers)
 *
 * Build: g++ -std=c++17 -O2 SapiRecorderBalanced.cpp -o SapiRecorder -lSDL2 -lSDL2_ttf -pthread
 */

#include "SapiRecorderBalanced.hpp"
#include "PrecomputedPaths.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <ctime>

namespace fs = std::filesystem;

// =============================================================================
// JSON WRITER (Simple implementation without external library)
// =============================================================================

class JsonWriter {
public:
    static std::string escapeString(const std::string& s) {
        std::ostringstream oss;
        for (char c : s) {
            switch (c) {
                case '"': oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default: oss << c; break;
            }
        }
        return oss.str();
    }

    static std::string segmentToJson(const SegmentData& seg) {
        std::ostringstream json;
        json << std::fixed << std::setprecision(6);

        json << "{\n";
        json << "  \"segment_id\": " << seg.segment_id << ",\n";
        json << "  \"timestamp_start\": " << seg.timestamp_start << ",\n";
        json << "  \"timestamp_end\": " << seg.timestamp_end << ",\n";
        json << "  \"duration_ms\": " << seg.duration_ms << ",\n";

        // Positions
        json << "  \"A\": {\"x\": " << seg.dot_A.x << ", \"y\": " << seg.dot_A.y << "},\n";
        json << "  \"B\": {\"x\": " << seg.dot_B.x << ", \"y\": " << seg.dot_B.y << "},\n";
        json << "  \"C\": {\"x\": " << seg.dot_C.x << ", \"y\": " << seg.dot_C.y << "},\n";

        // AB segment
        json << "  \"AB\": {\n";
        json << "    \"distance\": " << seg.ab_distance << ",\n";
        json << "    \"distance_group\": " << seg.ab_distance_group << ",\n";
        json << "    \"distance_name\": \"" << seg.ab_distance_name << "\",\n";
        json << "    \"angle_deg\": " << seg.ab_angle_deg << ",\n";
        json << "    \"orientation\": \"" << seg.ab_orientation << "\",\n";
        json << "    \"orientation_id\": " << seg.ab_orientation_id << "\n";
        json << "  },\n";

        // BC segment
        json << "  \"BC\": {\n";
        json << "    \"distance\": " << seg.bc_distance << ",\n";
        json << "    \"distance_group\": " << seg.bc_distance_group << ",\n";
        json << "    \"distance_name\": \"" << seg.bc_distance_name << "\",\n";
        json << "    \"angle_deg\": " << seg.bc_angle_deg << ",\n";
        json << "    \"orientation\": \"" << seg.bc_orientation << "\",\n";
        json << "    \"orientation_id\": " << seg.bc_orientation_id << "\n";
        json << "  },\n";

        // Turn
        json << "  \"turn\": {\n";
        json << "    \"angle_deg\": " << seg.turn_angle_deg << ",\n";
        json << "    \"category\": \"" << seg.turn_category << "\"\n";
        json << "  },\n";

        // Trajectory
        json << "  \"trajectory\": {\n";
        json << "    \"length\": " << seg.trajectory.size() << ",\n";

        // X coordinates
        json << "    \"x\": [";
        for (size_t i = 0; i < seg.trajectory.size(); ++i) {
            if (i > 0) json << ", ";
            json << seg.trajectory[i].x;
        }
        json << "],\n";

        // Y coordinates
        json << "    \"y\": [";
        for (size_t i = 0; i < seg.trajectory.size(); ++i) {
            if (i > 0) json << ", ";
            json << seg.trajectory[i].y;
        }
        json << "],\n";

        // Timestamps
        json << "    \"timestamps\": [";
        for (size_t i = 0; i < seg.trajectory.size(); ++i) {
            if (i > 0) json << ", ";
            json << seg.trajectory[i].timestamp;
        }
        json << "]\n";

        json << "  }\n";
        json << "}\n";

        return json.str();
    }
};

// =============================================================================
// COMPUTATION THREAD - Segment Metadata Processing
// =============================================================================

class ComputationThread {
public:
    ComputationThread(ThreadStats& stats)
        : stats_(stats), running_(false) {}

    ~ComputationThread() {
        stop();
    }

    void start() {
        running_ = true;
        thread_ = std::thread(&ComputationThread::run, this);
        ThreadUtils::setThreadName("Compute");
        std::cout << "[ComputationThread] Started\n";
    }

    void stop() {
        running_ = false;
        input_queue_.shutdown();
        if (thread_.joinable()) {
            thread_.join();
        }
        std::cout << "[ComputationThread] Stopped (processed: " << stats_.segments_computed << ")\n";
    }

    void enqueue(RawSegmentData raw) {
        input_queue_.push(std::move(raw));
    }

    void setOutputCallback(ComputationCompleteCallback cb) {
        output_callback_ = std::move(cb);
    }

    size_t pendingCount() const {
        return input_queue_.size();
    }

private:
    void run() {
        while (running_) {
            RawSegmentData raw;
            if (input_queue_.pop(raw, 50)) {
                auto start_time = std::chrono::high_resolution_clock::now();

                SegmentData segment = computeSegment(raw);

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                stats_.avg_compute_time_us = duration.count();
                stats_.segments_computed++;

                if (output_callback_) {
                    output_callback_(std::move(segment));
                }
            }
        }

        // Drain remaining
        RawSegmentData raw;
        while (input_queue_.pop(raw, 0)) {
            SegmentData segment = computeSegment(raw);
            if (output_callback_) {
                output_callback_(std::move(segment));
            }
        }
    }

    SegmentData computeSegment(const RawSegmentData& raw) {
        SegmentData seg;

        // Copy basic info
        seg.segment_id = raw.segment_id;
        seg.timestamp_start = raw.timestamp_start;
        seg.timestamp_end = raw.timestamp_end;
        seg.duration_ms = raw.timestamp_end - raw.timestamp_start;

        seg.dot_A = raw.dot_A;
        seg.dot_B = raw.dot_B;
        seg.dot_C = raw.dot_C;

        // Compute AB metrics
        double vec_AB_x = raw.dot_B.x - raw.dot_A.x;
        double vec_AB_y = raw.dot_B.y - raw.dot_A.y;
        seg.ab_distance = std::hypot(vec_AB_x, vec_AB_y);
        seg.ab_angle_deg = Helpers::rad2deg(std::atan2(vec_AB_y, vec_AB_x));
        seg.ab_orientation = Helpers::getOrientationFromAngle(seg.ab_angle_deg);
        seg.ab_orientation_id = Helpers::getOrientationId(seg.ab_orientation);
        auto [ab_group, ab_name] = Helpers::getDistanceGroup(seg.ab_distance);
        seg.ab_distance_group = ab_group;
        seg.ab_distance_name = ab_name;

        // Compute BC metrics
        double vec_BC_x = raw.dot_C.x - raw.dot_B.x;
        double vec_BC_y = raw.dot_C.y - raw.dot_B.y;
        seg.bc_distance = std::hypot(vec_BC_x, vec_BC_y);
        seg.bc_angle_deg = Helpers::rad2deg(std::atan2(vec_BC_y, vec_BC_x));
        seg.bc_orientation = Helpers::getOrientationFromAngle(seg.bc_angle_deg);
        seg.bc_orientation_id = Helpers::getOrientationId(seg.bc_orientation);
        auto [bc_group, bc_name] = Helpers::getDistanceGroup(seg.bc_distance);
        seg.bc_distance_group = bc_group;
        seg.bc_distance_name = bc_name;

        // Compute turn
        seg.turn_angle_deg = Helpers::angleBetweenVectors(vec_AB_x, vec_AB_y, vec_BC_x, vec_BC_y);
        seg.turn_category = Helpers::getTurnCategory(seg.turn_angle_deg);

        // Copy trajectory
        seg.trajectory = raw.trajectory;
        seg.computed = true;

        return seg;
    }

    ThreadStats& stats_;
    std::atomic<bool> running_;
    std::thread thread_;
    ThreadSafeQueue<RawSegmentData> input_queue_;
    ComputationCompleteCallback output_callback_;
};

// =============================================================================
// FILE I/O THREAD POOL - Parallel JSON Saving
// =============================================================================

class FileIOThreadPool {
public:
    FileIOThreadPool(const std::string& segments_dir, size_t num_workers, ThreadStats& stats)
        : segments_dir_(segments_dir), stats_(stats),
          pool_(num_workers, "FileIO") {}

    ~FileIOThreadPool() {
        stop();
    }

    void start() {
        pool_.start();
        std::cout << "[FileIOThreadPool] Started with " << pool_.workerCount() << " workers\n";
    }

    void stop() {
        pool_.stop();
        std::cout << "[FileIOThreadPool] Stopped (saved: " << stats_.files_saved << " files)\n";
    }

    void enqueueSegment(SegmentData segment) {
        pool_.submit([this, seg = std::move(segment)]() mutable {
            saveSegment(seg);
        });
    }

    size_t pendingCount() const {
        return pool_.pendingTasks();
    }

private:
    void saveSegment(const SegmentData& segment) {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::ostringstream filename;
        filename << "segment_" << std::setfill('0') << std::setw(4) << segment.segment_id << ".json";

        fs::path filepath = fs::path(segments_dir_) / filename.str();

        std::ofstream file(filepath);
        if (file.is_open()) {
            file << JsonWriter::segmentToJson(segment);
            file.close();
            stats_.files_saved++;
        } else {
            std::cerr << "[FileIOThreadPool] Failed to save: " << filepath << "\n";
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        stats_.avg_save_time_us = duration.count();
    }

    std::string segments_dir_;
    ThreadStats& stats_;
    ThreadPool<std::function<void()>> pool_;
};

// =============================================================================
// EVENT PROCESSING THREAD - Decoupled Event Handling
// =============================================================================

class EventProcessingThread {
public:
    using EventHandler = std::function<void(const MouseEvent&)>;

    EventProcessingThread(ThreadStats& stats)
        : stats_(stats), running_(false) {}

    ~EventProcessingThread() {
        stop();
    }

    void start() {
        running_ = true;
        thread_ = std::thread(&EventProcessingThread::run, this);
        ThreadUtils::setThreadName("Events");
        std::cout << "[EventProcessingThread] Started\n";
    }

    void stop() {
        running_ = false;
        event_queue_.shutdown();
        if (thread_.joinable()) {
            thread_.join();
        }
        std::cout << "[EventProcessingThread] Stopped (processed: " << stats_.events_processed << ")\n";
    }

    void pushEvent(MouseEvent event) {
        event_queue_.push(std::move(event));
    }

    void setHandler(EventHandler handler) {
        handler_ = std::move(handler);
    }

    size_t pendingCount() const {
        return event_queue_.size();
    }

private:
    void run() {
        while (running_) {
            MouseEvent event;
            if (event_queue_.pop(event, 20)) {
                if (handler_) {
                    handler_(event);
                    stats_.events_processed++;
                }
            }
        }

        // Drain remaining events
        MouseEvent event;
        while (event_queue_.pop(event, 0)) {
            if (handler_) {
                handler_(event);
                stats_.events_processed++;
            }
        }
    }

    ThreadStats& stats_;
    std::atomic<bool> running_;
    std::thread thread_;
    ThreadSafeQueue<MouseEvent> event_queue_;
    EventHandler handler_;
};

// =============================================================================
// SAMPLING THREAD (125Hz Mouse Polling) - Enhanced with Ring Buffer
// =============================================================================

class SamplingThread {
public:
    SamplingThread(ThreadStats& stats)
        : stats_(stats), running_(false), recording_enabled_(false), session_start_time_(0) {}

    ~SamplingThread() {
        stop();
    }

    void start(int64_t session_start) {
        session_start_time_ = session_start;
        running_ = true;
        thread_ = std::thread(&SamplingThread::run, this);
        ThreadUtils::setThreadName("Sampling");
        std::cout << "[SamplingThread] Started at " << Config::SAMPLE_RATE_HZ << "Hz\n";
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        std::cout << "[SamplingThread] Stopped (samples: " << stats_.samples_processed << ")\n";
    }

    void setRecordingEnabled(bool enabled) {
        recording_enabled_ = enabled;
    }

    void setDots(const Dot& A, const Dot& B) {
        std::lock_guard<std::mutex> lock(dots_mutex_);
        dot_A_ = A;
        dot_B_ = B;
    }

    void setCallbacks(MouseSampleCallback mouse_cb, EntryEventCallback entry_cb) {
        mouse_callback_ = std::move(mouse_cb);
        entry_callback_ = std::move(entry_cb);
    }

    void setWindowPosition(int x, int y) {
        window_x_.store(x, std::memory_order_relaxed);
        window_y_.store(y, std::memory_order_relaxed);
    }

    void resetEntryState() {
        inside_A_ = false;
        inside_B_ = false;
    }

    void setInsideA(bool value) { inside_A_ = value; }
    void setInsideB(bool value) { inside_B_ = value; }

private:
    void run() {
        const auto interval = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / Config::SAMPLE_RATE_HZ));
        auto next_time = std::chrono::high_resolution_clock::now();

        while (running_) {
            auto sample_start = std::chrono::high_resolution_clock::now();

            // Get mouse position
            int mouse_x, mouse_y;
            SDL_GetGlobalMouseState(&mouse_x, &mouse_y);

            // Convert to canvas coordinates
            int win_x = window_x_.load(std::memory_order_relaxed);
            int win_y = window_y_.load(std::memory_order_relaxed);

            int canvas_x = mouse_x - win_x;
            int canvas_y = mouse_y - win_y;

            // Check if within bounds
            if (canvas_x >= 0 && canvas_x <= Config::SCREEN_WIDTH &&
                canvas_y >= 0 && canvas_y <= Config::SCREEN_HEIGHT) {

                int64_t timestamp = Helpers::getCurrentTimeMs() - session_start_time_;

                // Check dot entry
                checkDotEntry(canvas_x, canvas_y, timestamp);

                // Record if enabled
                if (recording_enabled_ && mouse_callback_) {
                    mouse_callback_(timestamp, canvas_x, canvas_y);
                }

                stats_.samples_processed++;
            }

            // Measure latency
            auto sample_end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(sample_end - sample_start);
            stats_.avg_sample_latency_us = latency.count();

            // Precise timing for next sample
            next_time += interval;
            auto sleep_time = next_time - std::chrono::high_resolution_clock::now();

            if (sleep_time.count() > 0) {
                // Use hybrid sleep: OS sleep for most, then busy-wait for precision
                auto sleep_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(sleep_time);
                if (sleep_ns > std::chrono::microseconds(500)) {
                    std::this_thread::sleep_for(sleep_ns - std::chrono::microseconds(200));
                }
                // Busy-wait for final precision
                while (std::chrono::high_resolution_clock::now() < next_time) {
                    std::this_thread::yield();
                }
            } else {
                // We're behind, reset timing
                next_time = std::chrono::high_resolution_clock::now();
            }
        }
    }

    void checkDotEntry(int x, int y, int64_t timestamp) {
        if (timestamp - last_trigger_time_ < Config::MIN_TRIGGER_INTERVAL_MS) {
            return;
        }

        Dot A, B;
        {
            std::lock_guard<std::mutex> lock(dots_mutex_);
            A = dot_A_;
            B = dot_B_;
        }

        // Check A entry (start recording)
        if (!recording_enabled_) {
            bool inside_A_now = Helpers::isInsideCircle(x, y, A.x, A.y, Config::DOT_RADIUS);

            if (inside_A_now && !inside_A_) {
                inside_A_ = true;
                last_trigger_time_ = timestamp;
                if (entry_callback_) {
                    entry_callback_(RecorderEvent::StartRecording, timestamp, x, y);
                }
            } else if (!inside_A_now) {
                inside_A_ = false;
            }
        }

        // Check B entry (complete segment)
        if (recording_enabled_) {
            bool inside_B_now = Helpers::isInsideCircle(x, y, B.x, B.y, Config::DOT_RADIUS);

            if (inside_B_now && !inside_B_) {
                inside_B_ = true;
                last_trigger_time_ = timestamp;
                if (entry_callback_) {
                    entry_callback_(RecorderEvent::CompleteSegment, timestamp, x, y);
                }
            } else if (!inside_B_now) {
                inside_B_ = false;
            }
        }
    }

    ThreadStats& stats_;
    std::atomic<bool> running_;
    std::atomic<bool> recording_enabled_;
    std::thread thread_;

    int64_t session_start_time_;
    int64_t last_trigger_time_ = 0;

    std::mutex dots_mutex_;
    Dot dot_A_;
    Dot dot_B_;

    std::atomic<int> window_x_{0};
    std::atomic<int> window_y_{0};

    std::atomic<bool> inside_A_{false};
    std::atomic<bool> inside_B_{false};

    MouseSampleCallback mouse_callback_;
    EntryEventCallback entry_callback_;
};

// =============================================================================
// BALANCED THREE-DOT RECORDER (Main Class) - Enhanced Threading
// =============================================================================

class BalancedThreeDotRecorder {
public:
    BalancedThreeDotRecorder()
        : window_(nullptr), renderer_(nullptr), font_(nullptr),
          is_active_(false), recording_enabled_(false),
          segments_recorded_(0), path_index_(0),
          session_start_time_(0), segment_start_time_(0) {

        // Select random path
        precomputed_path_ = &PrecomputedPaths::getRandomPath();
        target_segments_ = static_cast<int>(precomputed_path_->size());

        // Generate session ID
        session_id_ = Helpers::generateSessionId();

        // Create output directories
        output_dir_ = fs::path(Config::OUTPUT_DIR) / session_id_;
        segments_dir_ = output_dir_ / "segments";
        fs::create_directories(segments_dir_);

        printBanner();
    }

    ~BalancedThreeDotRecorder() {
        cleanup();
    }

    bool initialize() {
        // Initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL Init failed: " << SDL_GetError() << "\n";
            return false;
        }

        // Initialize SDL_ttf
        if (TTF_Init() < 0) {
            std::cerr << "TTF Init failed: " << TTF_GetError() << "\n";
            return false;
        }

        // Create window
        window_ = SDL_CreateWindow(
            "SapiAgent Balanced Recorder v8.1.0 (C++ Enhanced Threading)",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            Config::SCREEN_WIDTH, Config::SCREEN_HEIGHT,
            SDL_WINDOW_SHOWN
        );
        if (!window_) {
            std::cerr << "Window creation failed: " << SDL_GetError() << "\n";
            return false;
        }

        // Create renderer with VSync
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer_) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << "\n";
            return false;
        }

        // Load font
        loadFont();

        // Initialize thread components
        initializeThreads();

        return true;
    }

    void run() {
        // Show countdown
        showStartupCountdown(5);

        // Start session and all threads
        startSession();

        // Main render loop (Thread 1)
        SDL_Event event;
        auto last_stats_time = std::chrono::steady_clock::now();

        while (is_active_) {
            // Handle SDL events
            while (SDL_PollEvent(&event)) {
                handleSDLEvent(event);
            }

            // Update window position for sampling thread
            int wx, wy;
            SDL_GetWindowPosition(window_, &wx, &wy);
            sampling_thread_->setWindowPosition(wx, wy);

            // Process events from event thread
            processMainThreadEvents();

            // Render
            render();

            // Print stats every 5 seconds
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 5) {
                printThreadStats();
                last_stats_time = now;
            }
        }

        // Show completion
        showCompletionScreen();
        SDL_Delay(5000);
    }

private:
    void printBanner() {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "BALANCED THREE-DOT RECORDER v8.1.0 (C++ Enhanced Multi-Threading)\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << "Session ID: " << session_id_ << "\n";
        std::cout << "Precomputed path: " << target_segments_ << " segments\n";
        std::cout << "Hardware threads: " << ThreadUtils::getHardwareThreads() << "\n";
        std::cout << "\nThread Architecture (6 threads optimized for i5-10600K):\n";
        std::cout << "  [1] Main Thread      - SDL2 rendering & window management\n";
        std::cout << "  [2] Event Thread     - Decoupled event processing\n";
        std::cout << "  [3] Sampling Thread  - " << Config::SAMPLE_RATE_HZ << "Hz mouse polling\n";
        std::cout << "  [4] Compute Thread   - Segment metadata calculations\n";
        std::cout << "  [5-6] File I/O Pool  - " << Config::FILE_IO_WORKER_COUNT << " parallel JSON writers\n";
        std::cout << "\nInstructions:\n";
        std::cout << "  1. Pass through GREEN dot (A) - recording starts\n";
        std::cout << "  2. Move to BLUE dot (B) - see ORANGE dot (C) as NEXT target\n";
        std::cout << "  3. Pass through BLUE dot (B) - segment completes\n";
        std::cout << "  4. Press ESC to end early\n";
        std::cout << std::string(70, '=') << "\n\n";
    }

    void initializeThreads() {
        // Create thread components
        event_thread_ = std::make_unique<EventProcessingThread>(stats_);
        sampling_thread_ = std::make_unique<SamplingThread>(stats_);
        compute_thread_ = std::make_unique<ComputationThread>(stats_);
        file_io_pool_ = std::make_unique<FileIOThreadPool>(
            segments_dir_.string(), Config::FILE_IO_WORKER_COUNT, stats_);

        // Wire up callbacks

        // Event thread handles entry events
        event_thread_->setHandler([this](const MouseEvent& evt) {
            std::lock_guard<std::mutex> lock(main_event_mutex_);
            main_event_queue_.push(evt);
        });

        // Sampling thread sends events to event thread
        sampling_thread_->setCallbacks(
            [this](int64_t ts, int x, int y) { onMouseSample(ts, x, y); },
            [this](RecorderEvent e, int64_t ts, int x, int y) {
                event_thread_->pushEvent({ts, x, y, e});
            }
        );

        // Compute thread sends completed segments to file I/O
        compute_thread_->setOutputCallback([this](SegmentData seg) {
            file_io_pool_->enqueueSegment(std::move(seg));
        });

        std::cout << "[Main] All thread components initialized\n";
    }

    void startSession() {
        session_start_time_ = Helpers::getCurrentTimeMs();
        is_active_ = true;

        // Spawn initial dots
        spawnInitialDots();

        // Start all threads
        event_thread_->start();
        sampling_thread_->start(session_start_time_);
        compute_thread_->start();
        file_io_pool_->start();

        // Configure sampling thread
        sampling_thread_->setDots(dot_A_, dot_B_);

        std::cout << "[Main] Session started - all threads running\n";
    }

    void endSession() {
        is_active_ = false;
        recording_enabled_ = false;
        sampling_thread_->setRecordingEnabled(false);

        // Stop threads in order
        sampling_thread_->stop();
        event_thread_->stop();
        compute_thread_->stop();
        file_io_pool_->stop();

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "SESSION COMPLETE\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << "  Segments recorded: " << segments_recorded_ << "/" << target_segments_ << "\n";
        std::cout << "  Output directory: " << output_dir_ << "\n";
        printThreadStats();
    }

    void printThreadStats() {
        std::cout << "\n[Thread Statistics]\n";
        std::cout << "  Samples processed:  " << stats_.samples_processed << "\n";
        std::cout << "  Events processed:   " << stats_.events_processed << "\n";
        std::cout << "  Segments computed:  " << stats_.segments_computed << "\n";
        std::cout << "  Files saved:        " << stats_.files_saved << "\n";
        std::cout << "  Avg sample latency: " << stats_.avg_sample_latency_us << " us\n";
        std::cout << "  Avg compute time:   " << stats_.avg_compute_time_us << " us\n";
        std::cout << "  Avg save time:      " << stats_.avg_save_time_us << " us\n";
        std::cout << "  Pending compute:    " << compute_thread_->pendingCount() << "\n";
        std::cout << "  Pending file I/O:   " << file_io_pool_->pendingCount() << "\n";
    }

    void cleanup() {
        if (sampling_thread_) sampling_thread_->stop();
        if (event_thread_) event_thread_->stop();
        if (compute_thread_) compute_thread_->stop();
        if (file_io_pool_) file_io_pool_->stop();

        if (font_) TTF_CloseFont(font_);
        if (renderer_) SDL_DestroyRenderer(renderer_);
        if (window_) SDL_DestroyWindow(window_);
        TTF_Quit();
        SDL_Quit();
    }

    void loadFont() {
        const char* font_paths[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            nullptr
        };

        for (int i = 0; font_paths[i] != nullptr; ++i) {
            font_ = TTF_OpenFont(font_paths[i], 16);
            if (font_) break;
        }

        if (!font_) {
            std::cerr << "Warning: Could not load any font\n";
        }
    }

    void showStartupCountdown(int seconds) {
        for (int i = seconds; i > 0; --i) {
            SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
            SDL_RenderClear(renderer_);

            renderText("BALANCED THREE-DOT RECORDER v8.1.0",
                       Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 - 100,
                       {255, 255, 255, 255}, true);

            renderText("Enhanced 6-Thread Architecture",
                       Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 - 60,
                       {100, 100, 100, 255}, true);

            std::ostringstream ss;
            ss << "Starting in " << i << "...";
            renderText(ss.str(), Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2,
                       {59, 130, 246, 255}, true);

            renderText("Following precomputed path for balanced coverage",
                       Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 + 60,
                       {128, 128, 128, 255}, true);

            SDL_RenderPresent(renderer_);
            SDL_Delay(1000);
        }
    }

    void showCompletionScreen() {
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);

        renderText("SESSION COMPLETE", Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 - 60,
                   {34, 197, 94, 255}, true);

        std::ostringstream ss;
        double pct = 100.0 * segments_recorded_ / target_segments_;
        ss << segments_recorded_ << "/" << target_segments_ << " segments ("
           << std::fixed << std::setprecision(0) << pct << "% coverage)";
        renderText(ss.str(), Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 + 20,
                   {255, 255, 255, 255}, true);

        ss.str("");
        ss << "Saved to: " << output_dir_.string();
        renderText(ss.str(), Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 + 80,
                   {128, 128, 128, 255}, true);

        SDL_RenderPresent(renderer_);
    }

    void spawnInitialDots() {
        const auto& first = (*precomputed_path_)[0];
        dot_A_.x = first.start_x;
        dot_A_.y = first.start_y;

        dot_B_ = getDotFromPath(0);

        if (precomputed_path_->size() > 1) {
            dot_C_ = getDotFromPath(1);
        } else {
            dot_C_ = dot_B_;
        }

        path_index_ = 0;
    }

    Dot getDotFromPath(size_t index) {
        if (index >= precomputed_path_->size()) {
            index = precomputed_path_->size() - 1;
        }

        const auto& seg = (*precomputed_path_)[index];
        auto it = Orientation::VECTORS.find(seg.orientation);
        double dx = it->second.x;
        double dy = it->second.y;

        Dot dot;
        dot.x = static_cast<int>(seg.start_x + dx * seg.distance);
        dot.y = static_cast<int>(seg.start_y + dy * seg.distance);
        dot.distance = seg.distance;
        dot.orientation = seg.orientation;

        auto [group, name] = Helpers::getDistanceGroup(seg.distance);
        dot.distance_group = group;
        dot.distance_name = name;

        return dot;
    }

    void shiftDots() {
        path_index_++;

        if (path_index_ >= precomputed_path_->size()) {
            return;
        }

        dot_A_.x = dot_B_.x;
        dot_A_.y = dot_B_.y;
        dot_B_ = dot_C_;

        if (path_index_ + 1 < precomputed_path_->size()) {
            dot_C_ = getDotFromPath(path_index_ + 1);
        } else {
            dot_C_ = dot_B_;
        }

        sampling_thread_->setDots(dot_A_, dot_B_);
    }

    void handleSDLEvent(const SDL_Event& event) {
        if (event.type == SDL_QUIT) {
            endSession();
        } else if (event.type == SDL_KEYDOWN) {
            if (event.key.keysym.sym == SDLK_ESCAPE) {
                endSession();
            }
        } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
            // Backup click handling
            int x = event.button.x;
            int y = event.button.y;
            int64_t timestamp = Helpers::getCurrentTimeMs() - session_start_time_;

            if (!recording_enabled_ && Helpers::isInsideCircle(x, y, dot_A_.x, dot_A_.y, Config::DOT_RADIUS)) {
                startRecording(timestamp, x, y);
            } else if (recording_enabled_ && Helpers::isInsideCircle(x, y, dot_B_.x, dot_B_.y, Config::DOT_RADIUS)) {
                completeSegment(timestamp, x, y);
            }
        }
    }

    void processMainThreadEvents() {
        std::lock_guard<std::mutex> lock(main_event_mutex_);
        while (!main_event_queue_.empty()) {
            MouseEvent evt = main_event_queue_.front();
            main_event_queue_.pop();

            if (evt.event_type == RecorderEvent::StartRecording) {
                startRecording(evt.timestamp, evt.x, evt.y);
            } else if (evt.event_type == RecorderEvent::CompleteSegment) {
                completeSegment(evt.timestamp, evt.x, evt.y);
            }
        }
    }

    void onMouseSample(int64_t timestamp, int x, int y) {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        current_trajectory_.push_back({timestamp, x, y});
    }

    void startRecording(int64_t timestamp, int x, int y) {
        if (recording_enabled_) return;

        recording_enabled_ = true;
        segment_start_time_ = timestamp;

        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            current_trajectory_.clear();
            current_trajectory_.push_back({timestamp, x, y});
        }

        sampling_thread_->setRecordingEnabled(true);
        std::cout << "[Recording] Started segment " << (segments_recorded_ + 1) << "\n";
    }

    void completeSegment(int64_t timestamp, int x, int y) {
        if (!recording_enabled_) return;

        // Copy trajectory
        std::vector<TrajectoryPoint> trajectory;
        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            current_trajectory_.push_back({timestamp, x, y});
            trajectory = current_trajectory_;
        }

        // Create raw segment for computation thread
        RawSegmentData raw;
        raw.segment_id = segments_recorded_ + 1;
        raw.timestamp_start = segment_start_time_;
        raw.timestamp_end = timestamp;
        raw.dot_A = dot_A_;
        raw.dot_B = dot_B_;
        raw.dot_C = dot_C_;
        raw.trajectory = std::move(trajectory);

        // Send to computation thread (non-blocking)
        compute_thread_->enqueue(std::move(raw));

        segments_recorded_++;
        std::cout << "[Recording] Completed segment " << segments_recorded_ << "/" << target_segments_ << "\n";

        // Check if done
        if (segments_recorded_ >= target_segments_) {
            recording_enabled_ = false;
            sampling_thread_->setRecordingEnabled(false);
            endSession();
            return;
        }

        // Shift dots
        shiftDots();

        // Reset entry detection
        sampling_thread_->setInsideA(true);
        sampling_thread_->setInsideB(false);

        // Reset trajectory for next segment
        segment_start_time_ = timestamp;
        {
            std::lock_guard<std::mutex> lock(trajectory_mutex_);
            current_trajectory_.clear();
            current_trajectory_.push_back({timestamp, x, y});
        }
    }

    void render() {
        // Clear
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);

        // Draw connecting lines
        SDL_SetRenderDrawColor(renderer_, 51, 51, 51, 255);
        SDL_RenderDrawLine(renderer_, dot_A_.x, dot_A_.y, dot_B_.x, dot_B_.y);
        SDL_SetRenderDrawColor(renderer_, 34, 34, 34, 255);
        SDL_RenderDrawLine(renderer_, dot_B_.x, dot_B_.y, dot_C_.x, dot_C_.y);

        // Draw C (next target) - Orange
        drawFilledCircle(dot_C_.x, dot_C_.y, Config::DOT_RADIUS - 5, {249, 115, 22, 255});
        renderText("C (next)", dot_C_.x, dot_C_.y - Config::DOT_RADIUS - 10, {249, 115, 22, 255}, true);

        // Draw B (current target) - Blue
        drawFilledCircle(dot_B_.x, dot_B_.y, Config::DOT_RADIUS, {59, 130, 246, 255});
        std::ostringstream b_label;
        b_label << "B (" << dot_B_.distance << "px " << dot_B_.orientation << ")";
        renderText(b_label.str(), dot_B_.x, dot_B_.y - Config::DOT_RADIUS - 10, {59, 130, 246, 255}, true);

        // Draw A (origin) - Green
        SDL_Color a_color = recording_enabled_ ? SDL_Color{144, 238, 144, 255} : SDL_Color{34, 197, 94, 255};
        drawFilledCircle(dot_A_.x, dot_A_.y, Config::DOT_RADIUS, a_color);
        std::string a_label = recording_enabled_ ? "A (recording...)" : "A (pass through)";
        renderText(a_label, dot_A_.x, dot_A_.y - Config::DOT_RADIUS - 10, {34, 197, 94, 255}, true);

        // Draw progress
        std::ostringstream progress;
        progress << segments_recorded_ << "/" << target_segments_;
        renderText(progress.str(), Config::SCREEN_WIDTH / 2, 25, {128, 128, 128, 255}, true);

        // Draw thread status
        std::ostringstream thread_status;
        thread_status << "Threads: Sampling=" << stats_.samples_processed
                      << " Compute=" << stats_.segments_computed
                      << " Saved=" << stats_.files_saved;
        renderText(thread_status.str(), Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT - 25,
                   {80, 80, 80, 255}, true);

        SDL_RenderPresent(renderer_);
    }

    void drawFilledCircle(int cx, int cy, int radius, SDL_Color color) {
        SDL_SetRenderDrawColor(renderer_, color.r, color.g, color.b, color.a);
        for (int w = 0; w < radius * 2; w++) {
            for (int h = 0; h < radius * 2; h++) {
                int dx = radius - w;
                int dy = radius - h;
                if ((dx * dx + dy * dy) <= (radius * radius)) {
                    SDL_RenderDrawPoint(renderer_, cx + dx, cy + dy);
                }
            }
        }
    }

    void renderText(const std::string& text, int x, int y, SDL_Color color, bool centered = false) {
        if (!font_) return;

        SDL_Surface* surface = TTF_RenderText_Blended(font_, text.c_str(), color);
        if (!surface) return;

        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer_, surface);
        if (!texture) {
            SDL_FreeSurface(surface);
            return;
        }

        SDL_Rect dst;
        dst.w = surface->w;
        dst.h = surface->h;
        dst.x = centered ? x - dst.w / 2 : x;
        dst.y = centered ? y - dst.h / 2 : y;

        SDL_RenderCopy(renderer_, texture, nullptr, &dst);

        SDL_DestroyTexture(texture);
        SDL_FreeSurface(surface);
    }

    // SDL components
    SDL_Window* window_;
    SDL_Renderer* renderer_;
    TTF_Font* font_;

    // Session state
    std::string session_id_;
    fs::path output_dir_;
    fs::path segments_dir_;
    const std::vector<PathSegment>* precomputed_path_;
    int target_segments_;

    std::atomic<bool> is_active_;
    std::atomic<bool> recording_enabled_;
    std::atomic<int> segments_recorded_;
    size_t path_index_;

    int64_t session_start_time_;
    int64_t segment_start_time_;

    // Dots
    Dot dot_A_;
    Dot dot_B_;
    Dot dot_C_;

    // Trajectory (protected by mutex)
    std::mutex trajectory_mutex_;
    std::vector<TrajectoryPoint> current_trajectory_;

    // Main thread event queue (from event processing thread)
    std::mutex main_event_mutex_;
    std::queue<MouseEvent> main_event_queue_;

    // Thread components
    std::unique_ptr<EventProcessingThread> event_thread_;
    std::unique_ptr<SamplingThread> sampling_thread_;
    std::unique_ptr<ComputationThread> compute_thread_;
    std::unique_ptr<FileIOThreadPool> file_io_pool_;

    // Thread statistics
    ThreadStats stats_;
};

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    std::cout << "SapiAgent Balanced Recorder v8.1.0\n";
    std::cout << "Enhanced 6-Thread Architecture for Intel i5-10600K\n";
    std::cout << "========================================================\n\n";

    BalancedThreeDotRecorder recorder;

    if (!recorder.initialize()) {
        std::cerr << "Failed to initialize recorder\n";
        return 1;
    }

    recorder.run();

    return 0;
}
