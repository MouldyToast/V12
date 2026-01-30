/**
 * SapiAgent Three-Dot Flow Recorder - BALANCED VERSION (C++)
 * Version: 8.0.0 - PRECOMPUTED BALANCED PATH
 *
 * Multi-threaded C++ implementation with dedicated threads for:
 * - Main Thread: Window management, rendering, event handling
 * - Sampling Thread: 125Hz mouse position polling
 * - File I/O Thread: Asynchronous segment JSON saving
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
// FILE I/O THREAD
// =============================================================================

class FileIOThread {
public:
    FileIOThread(const std::string& segments_dir)
        : segments_dir_(segments_dir), running_(false) {}

    ~FileIOThread() {
        stop();
    }

    void start() {
        running_ = true;
        thread_ = std::thread(&FileIOThread::run, this);
        std::cout << "[FileIOThread] Started\n";
    }

    void stop() {
        running_ = false;
        save_queue_.shutdown();
        if (thread_.joinable()) {
            thread_.join();
        }
        std::cout << "[FileIOThread] Stopped\n";
    }

    void enqueueSegment(const SegmentData& segment) {
        save_queue_.push(segment);
    }

    size_t pendingCount() const {
        return save_queue_.size();
    }

private:
    void run() {
        while (running_) {
            SegmentData segment;
            if (save_queue_.pop(segment, 100)) {
                saveSegment(segment);
            }
        }

        // Drain remaining segments
        SegmentData segment;
        while (save_queue_.pop(segment, 0)) {
            saveSegment(segment);
        }
    }

    void saveSegment(const SegmentData& segment) {
        std::ostringstream filename;
        filename << "segment_" << std::setfill('0') << std::setw(4) << segment.segment_id << ".json";

        fs::path filepath = fs::path(segments_dir_) / filename.str();

        std::ofstream file(filepath);
        if (file.is_open()) {
            file << JsonWriter::segmentToJson(segment);
            file.close();
        } else {
            std::cerr << "[FileIOThread] Failed to save: " << filepath << "\n";
        }
    }

    std::string segments_dir_;
    std::atomic<bool> running_;
    std::thread thread_;
    ThreadSafeQueue<SegmentData> save_queue_;
};

// =============================================================================
// SAMPLING THREAD (125Hz Mouse Polling)
// =============================================================================

class SamplingThread {
public:
    using MouseCallback = std::function<void(int64_t, int, int)>;
    using EntryCallback = std::function<void(RecorderEvent, int64_t, int, int)>;

    SamplingThread()
        : running_(false), recording_enabled_(false), session_start_time_(0) {}

    ~SamplingThread() {
        stop();
    }

    void start(int64_t session_start) {
        session_start_time_ = session_start;
        running_ = true;
        thread_ = std::thread(&SamplingThread::run, this);
        std::cout << "[SamplingThread] Started at " << Config::SAMPLE_RATE_HZ << "Hz\n";
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        std::cout << "[SamplingThread] Stopped\n";
    }

    void setRecordingEnabled(bool enabled) {
        recording_enabled_ = enabled;
    }

    void setDots(const Dot& A, const Dot& B) {
        std::lock_guard<std::mutex> lock(dots_mutex_);
        dot_A_ = A;
        dot_B_ = B;
    }

    void setCallbacks(MouseCallback mouse_cb, EntryCallback entry_cb) {
        mouse_callback_ = mouse_cb;
        entry_callback_ = entry_cb;
    }

    void setWindowPosition(int x, int y) {
        std::lock_guard<std::mutex> lock(window_mutex_);
        window_x_ = x;
        window_y_ = y;
    }

    // Reset entry detection state
    void resetEntryState() {
        inside_A_ = false;
        inside_B_ = false;
    }

    void setInsideA(bool value) { inside_A_ = value; }
    void setInsideB(bool value) { inside_B_ = value; }

private:
    void run() {
        const double interval_sec = 1.0 / Config::SAMPLE_RATE_HZ;
        auto next_time = std::chrono::high_resolution_clock::now();

        while (running_) {
            auto now = std::chrono::high_resolution_clock::now();

            // Get mouse position
            int mouse_x, mouse_y;
            SDL_GetGlobalMouseState(&mouse_x, &mouse_y);

            // Convert to canvas coordinates
            int win_x, win_y;
            {
                std::lock_guard<std::mutex> lock(window_mutex_);
                win_x = window_x_;
                win_y = window_y_;
            }

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
            }

            // Sleep until next sample
            next_time += std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
                std::chrono::duration<double>(interval_sec));

            auto sleep_time = next_time - std::chrono::high_resolution_clock::now();
            if (sleep_time.count() > 0) {
                std::this_thread::sleep_for(sleep_time);
            } else {
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

    std::atomic<bool> running_;
    std::atomic<bool> recording_enabled_;
    std::thread thread_;

    int64_t session_start_time_;
    int64_t last_trigger_time_ = 0;

    std::mutex dots_mutex_;
    Dot dot_A_;
    Dot dot_B_;

    std::mutex window_mutex_;
    int window_x_ = 0;
    int window_y_ = 0;

    std::atomic<bool> inside_A_{false};
    std::atomic<bool> inside_B_{false};

    MouseCallback mouse_callback_;
    EntryCallback entry_callback_;
};

// =============================================================================
// BALANCED THREE-DOT RECORDER (Main Class)
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

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "BALANCED THREE-DOT RECORDER v8.0.0 (C++)\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "Session ID: " << session_id_ << "\n";
        std::cout << "Precomputed path: " << target_segments_ << " segments\n";
        std::cout << "Coverage: 96 (dist x orient) combos x 2 = 192 segments\n";
        std::cout << "Turn categories: 7 types, ~27-28 each\n";
        std::cout << "\nThreads:\n";
        std::cout << "  - Main Thread: Rendering & Events\n";
        std::cout << "  - Sampling Thread: " << Config::SAMPLE_RATE_HZ << "Hz mouse polling\n";
        std::cout << "  - File I/O Thread: Async JSON saving\n";
        std::cout << "\nInstructions:\n";
        std::cout << "  1. Pass through GREEN dot (A) - recording starts\n";
        std::cout << "  2. Move to BLUE dot (B) - see ORANGE dot (C) as NEXT target\n";
        std::cout << "  3. Pass through BLUE dot (B) - segment completes\n";
        std::cout << "  4. Press ESC to end early\n";
        std::cout << std::string(60, '=') << "\n";
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
            "SapiAgent Balanced Recorder v8.0.0 (C++)",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            Config::SCREEN_WIDTH, Config::SCREEN_HEIGHT,
            SDL_WINDOW_SHOWN
        );
        if (!window_) {
            std::cerr << "Window creation failed: " << SDL_GetError() << "\n";
            return false;
        }

        // Create renderer
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer_) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << "\n";
            return false;
        }

        // Load font (try common system fonts)
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
            std::cerr << "Warning: Could not load any font, text will not display\n";
        }

        // Initialize file I/O thread
        file_io_thread_ = std::make_unique<FileIOThread>(segments_dir_.string());
        file_io_thread_->start();

        // Initialize sampling thread
        sampling_thread_ = std::make_unique<SamplingThread>();
        sampling_thread_->setCallbacks(
            [this](int64_t ts, int x, int y) { this->onMouseSample(ts, x, y); },
            [this](RecorderEvent e, int64_t ts, int x, int y) { this->onEntryEvent(e, ts, x, y); }
        );

        return true;
    }

    void run() {
        // Show countdown
        showStartupCountdown(5);

        // Start session
        startSession();

        // Main event loop
        SDL_Event event;
        while (is_active_) {
            while (SDL_PollEvent(&event)) {
                handleEvent(event);
            }

            // Update window position for sampling thread
            int wx, wy;
            SDL_GetWindowPosition(window_, &wx, &wy);
            sampling_thread_->setWindowPosition(wx, wy);

            // Render
            render();

            // Cap frame rate
            SDL_Delay(16);  // ~60 FPS
        }

        // Show completion
        showCompletionScreen();
        SDL_Delay(5000);
    }

private:
    void cleanup() {
        if (sampling_thread_) {
            sampling_thread_->stop();
        }
        if (file_io_thread_) {
            file_io_thread_->stop();
        }

        if (font_) TTF_CloseFont(font_);
        if (renderer_) SDL_DestroyRenderer(renderer_);
        if (window_) SDL_DestroyWindow(window_);
        TTF_Quit();
        SDL_Quit();
    }

    void showStartupCountdown(int seconds) {
        for (int i = seconds; i > 0; --i) {
            // Clear screen
            SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
            SDL_RenderClear(renderer_);

            // Render countdown text
            renderText("BALANCED THREE-DOT RECORDER", Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 - 80,
                      {255, 255, 255, 255}, true);

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

    void startSession() {
        session_start_time_ = Helpers::getCurrentTimeMs();
        is_active_ = true;

        // Spawn initial dots
        spawnInitialDots();

        // Start sampling thread
        sampling_thread_->start(session_start_time_);
        sampling_thread_->setDots(dot_A_, dot_B_);
    }

    void endSession() {
        is_active_ = false;
        recording_enabled_ = false;
        sampling_thread_->setRecordingEnabled(false);

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "SESSION COMPLETE\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  Segments recorded: " << segments_recorded_ << "/" << target_segments_ << "\n";
        std::cout << "  Output directory: " << output_dir_ << "\n";
    }

    void showCompletionScreen() {
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);

        renderText("SESSION COMPLETE", Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 - 60,
                  {34, 197, 94, 255}, true);

        std::ostringstream ss;
        double pct = 100.0 * segments_recorded_ / target_segments_;
        ss << segments_recorded_ << "/" << target_segments_ << " segments (" << std::fixed << std::setprecision(0) << pct << "% coverage)";
        renderText(ss.str(), Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 + 20,
                  {255, 255, 255, 255}, true);

        ss.str("");
        ss << "Saved to: " << output_dir_.string();
        renderText(ss.str(), Config::SCREEN_WIDTH / 2, Config::SCREEN_HEIGHT / 2 + 80,
                  {128, 128, 128, 255}, true);

        SDL_RenderPresent(renderer_);
    }

    void spawnInitialDots() {
        // A starts at first position in path
        const auto& first = (*precomputed_path_)[0];
        dot_A_.x = first.start_x;
        dot_A_.y = first.start_y;

        // B is the target for first segment
        dot_B_ = getDotFromPath(0);

        // C is the target for second segment
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

        // Compute target position
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

        // A becomes current B position
        dot_A_.x = dot_B_.x;
        dot_A_.y = dot_B_.y;

        // B becomes current C
        dot_B_ = dot_C_;

        // C becomes next target
        if (path_index_ + 1 < precomputed_path_->size()) {
            dot_C_ = getDotFromPath(path_index_ + 1);
        } else {
            dot_C_ = dot_B_;
        }

        // Update sampling thread
        sampling_thread_->setDots(dot_A_, dot_B_);
    }

    void handleEvent(const SDL_Event& event) {
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

    // Called from sampling thread
    void onMouseSample(int64_t timestamp, int x, int y) {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        current_trajectory_.push_back({timestamp, x, y});
    }

    // Called from sampling thread (via event queue for thread safety)
    void onEntryEvent(RecorderEvent event, int64_t timestamp, int x, int y) {
        // Queue event for main thread
        std::lock_guard<std::mutex> lock(event_mutex_);
        pending_events_.push({timestamp, x, y, event});
    }

    void processPendingEvents() {
        std::lock_guard<std::mutex> lock(event_mutex_);
        while (!pending_events_.empty()) {
            MouseEvent evt = pending_events_.front();
            pending_events_.pop();

            if (evt.event_type == RecorderEvent::StartRecording) {
                startRecording(evt.timestamp, evt.x, evt.y);
            } else if (evt.event_type == RecorderEvent::CompleteSegment) {
                completeSegment(evt.timestamp, evt.x, evt.y);
            }
        }
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

        // Build segment data
        SegmentData segment;
        segment.segment_id = segments_recorded_ + 1;
        segment.timestamp_start = segment_start_time_;
        segment.timestamp_end = timestamp;
        segment.duration_ms = timestamp - segment_start_time_;

        segment.dot_A = dot_A_;
        segment.dot_B = dot_B_;
        segment.dot_C = dot_C_;

        // Compute AB metrics
        double vec_AB_x = dot_B_.x - dot_A_.x;
        double vec_AB_y = dot_B_.y - dot_A_.y;
        segment.ab_distance = std::hypot(vec_AB_x, vec_AB_y);
        segment.ab_angle_deg = Helpers::rad2deg(std::atan2(vec_AB_y, vec_AB_x));
        segment.ab_orientation = Helpers::getOrientationFromAngle(segment.ab_angle_deg);
        segment.ab_orientation_id = Helpers::getOrientationId(segment.ab_orientation);
        auto [ab_group, ab_name] = Helpers::getDistanceGroup(segment.ab_distance);
        segment.ab_distance_group = ab_group;
        segment.ab_distance_name = ab_name;

        // Compute BC metrics
        double vec_BC_x = dot_C_.x - dot_B_.x;
        double vec_BC_y = dot_C_.y - dot_B_.y;
        segment.bc_distance = std::hypot(vec_BC_x, vec_BC_y);
        segment.bc_angle_deg = Helpers::rad2deg(std::atan2(vec_BC_y, vec_BC_x));
        segment.bc_orientation = Helpers::getOrientationFromAngle(segment.bc_angle_deg);
        segment.bc_orientation_id = Helpers::getOrientationId(segment.bc_orientation);
        auto [bc_group, bc_name] = Helpers::getDistanceGroup(segment.bc_distance);
        segment.bc_distance_group = bc_group;
        segment.bc_distance_name = bc_name;

        // Compute turn
        segment.turn_angle_deg = Helpers::angleBetweenVectors(vec_AB_x, vec_AB_y, vec_BC_x, vec_BC_y);
        segment.turn_category = Helpers::getTurnCategory(segment.turn_angle_deg);

        segment.trajectory = trajectory;

        // Enqueue for saving
        file_io_thread_->enqueueSegment(segment);

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

        // Reset entry detection but stay recording
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
        // Process pending events from sampling thread
        processPendingEvents();

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

        // Draw A (origin) - Green (lighter if recording)
        SDL_Color a_color = recording_enabled_ ? SDL_Color{144, 238, 144, 255} : SDL_Color{34, 197, 94, 255};
        drawFilledCircle(dot_A_.x, dot_A_.y, Config::DOT_RADIUS, a_color);
        std::string a_label = recording_enabled_ ? "A (recording...)" : "A (pass through)";
        renderText(a_label, dot_A_.x, dot_A_.y - Config::DOT_RADIUS - 10, {34, 197, 94, 255}, true);

        // Draw progress
        std::ostringstream progress;
        progress << segments_recorded_ << "/" << target_segments_;
        renderText(progress.str(), Config::SCREEN_WIDTH / 2, 25, {128, 128, 128, 255}, true);

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

    // Event queue from sampling thread
    std::mutex event_mutex_;
    std::queue<MouseEvent> pending_events_;

    // Thread components
    std::unique_ptr<SamplingThread> sampling_thread_;
    std::unique_ptr<FileIOThread> file_io_thread_;
};

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "SapiAgent Balanced Recorder v8.0.0 (C++ Multi-threaded)\n";
    std::cout << "========================================================\n\n";

    BalancedThreeDotRecorder recorder;

    if (!recorder.initialize()) {
        std::cerr << "Failed to initialize recorder\n";
        return 1;
    }

    recorder.run();

    return 0;
}
