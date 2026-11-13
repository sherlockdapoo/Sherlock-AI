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

// Sherlock's tiny character-level model, now trained from an external corpus.
// A trigram stores counts for: previous two characters -> next character.
// The browser downloads the corpus; all model updates happen inside this C++ WASM module.

static const int ALPHABET = 128;
static const int CONTEXTS = ALPHABET * ALPHABET;
static const uint32_t TRAIN_CAPACITY = 9u * 1024u * 1024u;
static const uint32_t VALIDATION_CAPACITY = 128u * 1024u;

static uint16_t counts[CONTEXTS * ALPHABET];
static uint32_t context_totals[CONTEXTS];
static uint8_t training_data[TRAIN_CAPACITY];
static uint8_t validation_data[VALIDATION_CAPACITY];
static uint32_t training_length = 0;
static uint32_t validation_length = 0;
static uint32_t training_cursor = 2;
static uint32_t seen_characters = 0;
static uint32_t training_steps = 0;
static int best_context = 0;
static uint32_t best_context_total = 0;
static uint32_t random_state = 0x51A7C0DEu;
static uint8_t input_buffer[256];
static uint8_t output_buffer[512];
static int output_length = 0;

static int safe_char(unsigned char c) { return c < ALPHABET ? (int)c : (int)'?'; }
static uint32_t next_random() {
    uint32_t x=random_state; x^=x<<13; x^=x>>17; x^=x<<5;
    random_state=x?x:0xA341316Cu; return random_state;
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
    training_cursor = 2;
}
__attribute__((visibility("default"))) void set_validation_length(int length) {
    if (length < 0) length = 0;
    if ((uint32_t)length > VALIDATION_CAPACITY) length = (int)VALIDATION_CAPACITY;
    validation_length = (uint32_t)length;
}
__attribute__((visibility("default"))) void reset_model() {
    for (uint32_t i=0;i<(uint32_t)(CONTEXTS*ALPHABET);++i) counts[i]=0;
    for (int i=0;i<CONTEXTS;++i) context_totals[i]=0;
    training_cursor=2; seen_characters=0; training_steps=0;
    best_context=0; best_context_total=0; random_state=0x51A7C0DEu;
}

__attribute__((visibility("default"))) int train_step(int character_budget) {
    if (training_length < 3) return 0;
    if (character_budget<1) character_budget=1;
    if (character_budget>200000) character_budget=200000;
    for (int i=0;i<character_budget;++i) {
        if (training_cursor>=training_length) training_cursor=2;
        int a=safe_char(training_data[training_cursor-2]);
        int b=safe_char(training_data[training_cursor-1]);
        int next=safe_char(training_data[training_cursor]);
        int context=a*ALPHABET+b;
        uint32_t slot=(uint32_t)context*ALPHABET+(uint32_t)next;
        if (counts[slot]<65535u) {
            ++counts[slot];
            ++context_totals[context];
        }
        if (context_totals[context]>best_context_total) {
            best_context_total=context_totals[context]; best_context=context;
        }
        ++training_cursor; ++seen_characters;
    }
    ++training_steps;
    return (int)seen_characters;
}

// Held-out next-character accuracy in tenths of one percent.
// Samples up to roughly 3,000 positions so checking the score stays responsive on iPad.
__attribute__((visibility("default"))) int accuracy_tenths() {
    if (validation_length < 3) return 0;
    uint32_t sample_count = validation_length > 60000u ? 3000u : validation_length;
    uint32_t stride = validation_length > 60000u ? validation_length / sample_count : 1u;
    if (stride < 1u) stride = 1u;
    int correct=0, total=0;
    for (uint32_t i=2;i<validation_length;i+=stride) {
        int a=safe_char(validation_data[i-2]);
        int b=safe_char(validation_data[i-1]);
        int actual=safe_char(validation_data[i]);
        int context=a*ALPHABET+b;
        uint32_t base=(uint32_t)context*ALPHABET;
        uint16_t highest=0; int prediction=' ';
        for (int c=0;c<ALPHABET;++c) {
            uint16_t value=counts[base+(uint32_t)c];
            if (value>highest) { highest=value; prediction=c; }
        }
        if (highest && prediction==actual) ++correct;
        ++total;
    }
    return total ? (correct*1000+total/2)/total : 0;
}

__attribute__((visibility("default"))) int characters_seen() { return (int)seen_characters; }
__attribute__((visibility("default"))) int steps_completed() { return (int)training_steps; }
__attribute__((visibility("default"))) int training_position() { return (int)training_cursor; }
__attribute__((visibility("default"))) int training_size() { return (int)training_length; }

__attribute__((visibility("default"))) int generate_text(int prompt_length,int requested_length) {
    if (prompt_length<0) prompt_length=0;
    if (prompt_length>240) prompt_length=240;
    if (requested_length<1) requested_length=1;
    if (requested_length>200) requested_length=200;
    output_length=0;
    for (int i=0;i<prompt_length && output_length<300;++i) output_buffer[output_length++]=input_buffer[i];
    if (seen_characters==0 || output_length==0) {
        const char fallback[]="Train me first.";
        for (int i=0;fallback[i] && output_length<300;++i) output_buffer[output_length++]=(uint8_t)fallback[i];
        return output_length;
    }
    int previous=output_length>=2?safe_char(output_buffer[output_length-2]):' ';
    int current=output_length>=1?safe_char(output_buffer[output_length-1]):' ';
    for (int step=0;step<requested_length && output_length<500;++step) {
        int context=previous*ALPHABET+current;
        uint32_t base=(uint32_t)context*ALPHABET, total=context_totals[context];
        if (total==0) {
            context=best_context; previous=context/ALPHABET; current=context%ALPHABET;
            base=(uint32_t)context*ALPHABET; total=context_totals[context];
        }
        if (total==0) break;
        uint32_t pick=next_random()%total; int chosen=' ';
        for (int c=0;c<ALPHABET;++c) {
            uint16_t value=counts[base+(uint32_t)c];
            if (pick<value) { chosen=c; break; }
            pick-=value;
        }
        output_buffer[output_length++]=(uint8_t)chosen;
        previous=current; current=chosen;
    }
    return output_length;
}
}