#include <stdint.h>

// Sherlock's tiny character-level language model.
// A trigram stores counts for: previous two characters -> next character.
// This model runs locally inside the browser's WebAssembly VM.

static const char TRAIN_TEXT[] =
    "sherlock is a small character language model. "
    "a language model learns patterns by reading examples of text. "
    "it studies which character often follows the two characters before it. "
    "when it has seen a pattern many times it can make a useful prediction. "
    "the quick brown fox jumps over the lazy dog. "
    "learning takes examples, practice, testing, and careful measurement. "
    "a computer can count patterns very quickly, but counting is not the same as understanding. "
    "sherlock runs locally in the browser and uses the ipad processor for its calculations. "
    "this experiment creates text one character at a time. "
    "the best next character is chosen from patterns found in the training examples. "
    "sometimes a prediction will be right and sometimes it will be wrong. "
    "a score measures how often the model predicts the next character in separate test text. "
    "good training data gives a model useful examples to learn from. "
    "small models are useful for experiments because their behaviour is easier to inspect. "
    "fast code can process many examples, while a good model still needs suitable data. "
    "the sky is blue, the grass is green, and the stars shine at night. "
    "we can explore science, maths, stories, computers, and the world around us. "
    "every sentence contains patterns in letters, spaces, and punctuation. "
    "sherlock learns statistical patterns rather than facts or human meaning. ";

static const char TEST_TEXT[] =
    "a little model reads text and predicts the next character. "
    "the stars are bright in the night sky. "
    "computers learn patterns from examples and use them to generate text. "
    "practice and testing help us measure a model carefully. ";

static const int ALPHABET = 128;
static const int CONTEXTS = ALPHABET * ALPHABET;
static uint16_t counts[CONTEXTS * ALPHABET];
static uint32_t context_totals[CONTEXTS];
static int training_cursor = 2;
static uint32_t seen_characters = 0;
static uint32_t training_steps = 0;
static int best_context = 0;
static uint32_t best_context_total = 0;
static uint32_t random_state = 0x51A7C0DEu;
static uint8_t input_buffer[256];
static uint8_t output_buffer[512];
static int output_length = 0;

static int safe_char(unsigned char c) { return c < ALPHABET ? (int)c : (int)'?'; }
static int train_length() { int n=0; while (TRAIN_TEXT[n]) ++n; return n; }
static int test_length() { int n=0; while (TEST_TEXT[n]) ++n; return n; }
static uint32_t next_random() {
    uint32_t x=random_state; x^=x<<13; x^=x>>17; x^=x<<5;
    random_state=x?x:0xA341316Cu; return random_state;
}

extern "C" {
__attribute__((visibility("default"))) int input_ptr() { return (int)(uintptr_t)input_buffer; }
__attribute__((visibility("default"))) int output_ptr() { return (int)(uintptr_t)output_buffer; }
__attribute__((visibility("default"))) int output_len() { return output_length; }

__attribute__((visibility("default"))) void reset_model() {
    for (uint32_t i=0;i<(uint32_t)(CONTEXTS*ALPHABET);++i) counts[i]=0;
    for (int i=0;i<CONTEXTS;++i) context_totals[i]=0;
    training_cursor=2; seen_characters=0; training_steps=0;
    best_context=0; best_context_total=0; random_state=0x51A7C0DEu;
}

__attribute__((visibility("default"))) int train_step(int character_budget) {
    const int length=train_length();
    if (character_budget<1) character_budget=1;
    if (character_budget>20000) character_budget=20000;
    for (int i=0;i<character_budget;++i) {
        if (training_cursor>=length) training_cursor=2;
        int a=safe_char((unsigned char)TRAIN_TEXT[training_cursor-2]);
        int b=safe_char((unsigned char)TRAIN_TEXT[training_cursor-1]);
        int next=safe_char((unsigned char)TRAIN_TEXT[training_cursor]);
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

// Returns held-out next-character prediction accuracy in tenths of one percent.
__attribute__((visibility("default"))) int accuracy_tenths() {
    int length=test_length(), correct=0, total=0;
    for (int i=2;i<length;++i) {
        int a=safe_char((unsigned char)TEST_TEXT[i-2]);
        int b=safe_char((unsigned char)TEST_TEXT[i-1]);
        int actual=safe_char((unsigned char)TEST_TEXT[i]);
        int context=a*ALPHABET+b;
        uint32_t base=(uint32_t)context*ALPHABET;
        uint16_t highest=0; int prediction=-1;
        for (int c=0;c<ALPHABET;++c) {
            uint16_t value=counts[base+(uint32_t)c];
            if (value>highest) { highest=value; prediction=c; }
        }
        if (prediction==actual) ++correct;
        ++total;
    }
    return total ? (correct*1000+total/2)/total : 0;
}

__attribute__((visibility("default"))) int characters_seen() { return (int)seen_characters; }
__attribute__((visibility("default"))) int steps_completed() { return (int)training_steps; }

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