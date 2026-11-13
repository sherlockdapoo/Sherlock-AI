#include <stdint.h>

extern "C" void *memset(void *destination, int value, unsigned long count) {
    uint8_t *out = (uint8_t *)destination;
    for (unsigned long i = 0; i < count; ++i) out[i] = (uint8_t)value;
    return destination;
}
extern "C" void *memcpy(void *destination, const void *source, unsigned long count) {
    uint8_t *out = (uint8_t *)destination;
    const uint8_t *in = (const uint8_t *)source;
    for (unsigned long i = 0; i < count; ++i) out[i] = in[i];
    return destination;
}

// Sherlock v0.9: a character-level 4-gram model with backoff.
// It uses the previous three characters when enough evidence exists,
// otherwise it backs off to the previous two characters.
// Training and prediction run locally in the browser's WebAssembly engine.

static const uint32_t ALPHABET = 128u;
static const uint32_t CONTEXT2_COUNT = ALPHABET * ALPHABET;
static const uint32_t TRAIN_CAPACITY = 9u * 1024u * 1024u;
static const uint32_t VALIDATION_CAPACITY = 128u * 1024u;
static const uint32_t CONTEXT3_SLOTS = 65536u;
static const uint32_t MAX_CONTEXT3_ENTRIES = 49152u; // Keep the hash table below 75% full.
static const uint32_t CONTEXT3_MASK = CONTEXT3_SLOTS - 1u;
static const uint32_t EMPTY_KEY = 0xFFFFFFFFu;
static const uint32_t MIN_CONTEXT3_COUNT = 3u;

struct Context3Entry {
    uint32_t key;
    uint32_t total;
    uint32_t counts[ALPHABET];
};

// Full-width counters avoid the old 65,535-count ceiling.
static uint32_t counts2[CONTEXT2_COUNT * ALPHABET];
static uint32_t totals2[CONTEXT2_COUNT];
// Sparse table: each occupied entry stores a 3-character context and its
// next-character counts. 65,536 slots keep memory bounded on mobile devices.
static Context3Entry contexts3[CONTEXT3_SLOTS];

// A compact best-validation checkpoint. It is updated only when the held-out
// score improves, and can be restored when the user stops training.
static uint32_t best_counts2[CONTEXT2_COUNT * ALPHABET];
static uint32_t best_totals2[CONTEXT2_COUNT];
static Context3Entry best_contexts3[CONTEXT3_SLOTS];

static uint8_t training_data[TRAIN_CAPACITY];
static uint8_t validation_data[VALIDATION_CAPACITY];
static uint32_t training_length = 0;
static uint32_t validation_length = 0;
static uint32_t training_cursor = 3;
static uint32_t seen_characters = 0;
static uint32_t training_steps = 0;
static uint32_t training_passes = 0;
static int best_context2 = 0;
static uint32_t best_context2_total = 0;
static int checkpoint_context2 = 0;
static uint32_t checkpoint_context2_total = 0;
static uint32_t context3_used = 0;
static uint32_t checkpoint_context3_used = 0;
static uint32_t checkpoint_accuracy = 0;
static int checkpoint_saved = 0;
static uint32_t random_state = 0x51A7C0DEu;
static uint8_t input_buffer[256];
static uint8_t output_buffer[512];

static int safe_char(unsigned char c) {
    return c < ALPHABET ? (int)c : (int)'?';
}

static uint32_t next_random() {
    uint32_t x = random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    random_state = x ? x : 0xA341316Cu;
    return random_state;
}

static uint32_t context3_key(int a, int b, int c) {
    return ((uint32_t)a * ALPHABET + (uint32_t)b) * ALPHABET + (uint32_t)c;
}

static Context3Entry *find_context3(uint32_t key, int create) {
    uint32_t slot = (key * 2654435761u) >> 16;
    for (uint32_t probe = 0; probe < CONTEXT3_SLOTS; ++probe) {
        Context3Entry *entry = &contexts3[(slot + probe) & CONTEXT3_MASK];
        if (entry->key == key) return entry;
        if (entry->key == EMPTY_KEY) {
            if (!create || context3_used >= MAX_CONTEXT3_ENTRIES) return 0;
            entry->key = key;
            entry->total = 0;
            for (uint32_t i = 0; i < ALPHABET; ++i) entry->counts[i] = 0;
            ++context3_used;
            return entry;
        }
    }
    return 0; // If full, the two-character backoff model still learns.
}

static int most_likely(const uint32_t *row) {
    uint32_t highest = 0;
    int prediction = ' ';
    for (uint32_t c = 0; c < ALPHABET; ++c) {
        if (row[c] > highest) {
            highest = row[c];
            prediction = (int)c;
        }
    }
    return highest ? prediction : -1;
}

static int sample_from_row(const uint32_t *row) {
    uint64_t total = 0;
    for (uint32_t c = 0; c < ALPHABET; ++c) total += row[c];
    if (!total) return -1;
    uint64_t pick = (uint64_t)next_random() % total;
    for (uint32_t c = 0; c < ALPHABET; ++c) {
        if (pick < row[c]) return (int)c;
        pick -= row[c];
    }
    return ' ';
}

static uint32_t score_validation() {
    if (validation_length < 4u) return 0;
    uint32_t sample_count = validation_length > 10000u ? 10000u : validation_length - 3u;
    uint32_t stride = validation_length > 10000u ? validation_length / sample_count : 1u;
    if (!stride) stride = 1u;

    uint32_t correct = 0;
    uint32_t total = 0;
    for (uint32_t i = 3; i < validation_length; i += stride) {
        int a = safe_char(validation_data[i - 3]);
        int b = safe_char(validation_data[i - 2]);
        int c = safe_char(validation_data[i - 1]);
        int actual = safe_char(validation_data[i]);
        int prediction = -1;

        Context3Entry *entry = find_context3(context3_key(a, b, c), 0);
        if (entry && entry->total >= MIN_CONTEXT3_COUNT)
            prediction = most_likely(entry->counts);

        if (prediction < 0) {
            int context2 = b * (int)ALPHABET + c;
            const uint32_t *row = &counts2[(uint32_t)context2 * ALPHABET];
            prediction = most_likely(row);
        }
        if (prediction == actual) ++correct;
        ++total;
    }
    return total ? (uint32_t)(((uint64_t)correct * 100000u + total / 2u) / total) : 0u;
}

static void save_best_checkpoint(uint32_t score) {
    if (checkpoint_saved && score <= checkpoint_accuracy) return;
    memcpy(best_counts2, counts2, sizeof(counts2));
    memcpy(best_totals2, totals2, sizeof(totals2));
    memcpy(best_contexts3, contexts3, sizeof(contexts3));
    checkpoint_context2 = best_context2;
    checkpoint_context2_total = best_context2_total;
    checkpoint_context3_used = context3_used;
    checkpoint_accuracy = score;
    checkpoint_saved = 1;
}

extern "C" {
__attribute__((visibility("default"))) int input_ptr() { return (int)(uintptr_t)input_buffer; }
__attribute__((visibility("default"))) int output_ptr() { return (int)(uintptr_t)output_buffer; }
__attribute__((visibility("default"))) int training_ptr() { return (int)(uintptr_t)training_data; }
__attribute__((visibility("default"))) int validation_ptr() { return (int)(uintptr_t)validation_data; }
__attribute__((visibility("default"))) int training_capacity() { return (int)TRAIN_CAPACITY; }
__attribute__((visibility("default"))) int validation_capacity() { return (int)VALIDATION_CAPACITY; }

__attribute__((visibility("default"))) void set_training_length(int length) {
    if (length < 0) length = 0;
    if ((uint32_t)length > TRAIN_CAPACITY) length = (int)TRAIN_CAPACITY;
    training_length = (uint32_t)length;
    training_cursor = 3;
}

__attribute__((visibility("default"))) void set_validation_length(int length) {
    if (length < 0) length = 0;
    if ((uint32_t)length > VALIDATION_CAPACITY) length = (int)VALIDATION_CAPACITY;
    validation_length = (uint32_t)length;
}

__attribute__((visibility("default"))) void reset_model() {
    memset(counts2, 0, sizeof(counts2));
    memset(totals2, 0, sizeof(totals2));
    for (uint32_t i = 0; i < CONTEXT3_SLOTS; ++i) contexts3[i].key = EMPTY_KEY;
    training_cursor = 3;
    seen_characters = 0;
    training_steps = 0;
    training_passes = 0;
    best_context2 = 0;
    best_context2_total = 0;
    context3_used = 0;
    checkpoint_context2 = 0;
    checkpoint_context2_total = 0;
    checkpoint_context3_used = 0;
    checkpoint_accuracy = 0;
    checkpoint_saved = 0;
    random_state = 0x51A7C0DEu;
}

__attribute__((visibility("default"))) int train_step(int character_budget) {
    if (training_length < 4u) return 0;
    if (character_budget < 1) character_budget = 1;
    if (character_budget > 200000) character_budget = 200000;

    for (int i = 0; i < character_budget; ++i) {
        int a = safe_char(training_data[training_cursor - 3]);
        int b = safe_char(training_data[training_cursor - 2]);
        int c = safe_char(training_data[training_cursor - 1]);
        int next = safe_char(training_data[training_cursor]);

        int context2 = b * (int)ALPHABET + c;
        uint32_t slot2 = (uint32_t)context2 * ALPHABET + (uint32_t)next;
        if (counts2[slot2] < 0xFFFFFFFFu) ++counts2[slot2];
        if (totals2[context2] < 0xFFFFFFFFu) ++totals2[context2];
        if (totals2[context2] > best_context2_total) {
            best_context2_total = totals2[context2];
            best_context2 = context2;
        }

        Context3Entry *entry = find_context3(context3_key(a, b, c), 1);
        if (entry) {
            if (entry->counts[next] < 0xFFFFFFFFu) ++entry->counts[next];
            if (entry->total < 0xFFFFFFFFu) ++entry->total;
        }

        ++training_cursor;
        ++seen_characters;
        if (training_cursor >= training_length) {
            training_cursor = 3;
            ++training_passes;
        }
    }
    ++training_steps;
    return (int)seen_characters;
}

// Fixed held-out accuracy, in thousandths of one percent. A new best model is
// snapshotted only when its validation score improves.
__attribute__((visibility("default"))) int accuracy_thousandths_percent() {
    uint32_t score = score_validation();
    save_best_checkpoint(score);
    return (int)score;
}

__attribute__((visibility("default"))) int best_model_accuracy_thousandths_percent() {
    return checkpoint_saved ? (int)checkpoint_accuracy : 0;
}

__attribute__((visibility("default"))) int has_best_model() { return checkpoint_saved; }

__attribute__((visibility("default"))) int restore_best_model() {
    if (!checkpoint_saved) return 0;
    memcpy(counts2, best_counts2, sizeof(counts2));
    memcpy(totals2, best_totals2, sizeof(totals2));
    memcpy(contexts3, best_contexts3, sizeof(contexts3));
    best_context2 = checkpoint_context2;
    best_context2_total = checkpoint_context2_total;
    context3_used = checkpoint_context3_used;
    return 1;
}

__attribute__((visibility("default"))) int characters_seen() { return (int)seen_characters; }
__attribute__((visibility("default"))) int steps_completed() { return (int)training_steps; }
__attribute__((visibility("default"))) int training_passes_completed() { return (int)training_passes; }
__attribute__((visibility("default"))) int training_position() { return (int)training_cursor; }
__attribute__((visibility("default"))) int training_size() { return (int)training_length; }

__attribute__((visibility("default"))) int generate_text(int prompt_length, int requested_length) {
    if (prompt_length < 0) prompt_length = 0;
    if (prompt_length > 240) prompt_length = 240;
    if (requested_length < 1) requested_length = 1;
    if (requested_length > 200) requested_length = 200;

    int output_length = 0;
    for (int i = 0; i < prompt_length && output_length < 300; ++i)
        output_buffer[output_length++] = input_buffer[i];

    if (seen_characters == 0 || output_length == 0) {
        const char fallback[] = "Train me first.";
        for (int i = 0; fallback[i] && output_length < 300; ++i)
            output_buffer[output_length++] = (uint8_t)fallback[i];
        return output_length;
    }

    for (int step = 0; step < requested_length && output_length < 500; ++step) {
        int chosen = -1;
        if (output_length >= 3) {
            int a = safe_char(output_buffer[output_length - 3]);
            int b = safe_char(output_buffer[output_length - 2]);
            int c = safe_char(output_buffer[output_length - 1]);
            Context3Entry *entry = find_context3(context3_key(a, b, c), 0);
            if (entry && entry->total >= MIN_CONTEXT3_COUNT)
                chosen = sample_from_row(entry->counts);
        }

        if (chosen < 0 && output_length >= 2) {
            int b = safe_char(output_buffer[output_length - 2]);
            int c = safe_char(output_buffer[output_length - 1]);
            int context2 = b * (int)ALPHABET + c;
            chosen = sample_from_row(&counts2[(uint32_t)context2 * ALPHABET]);
        }

        if (chosen < 0 && totals2[best_context2] > 0)
            chosen = sample_from_row(&counts2[(uint32_t)best_context2 * ALPHABET]);
        if (chosen < 0) break;
        output_buffer[output_length++] = (uint8_t)chosen;
    }
    return output_length;
}
}