#include "model_loader.hpp"
#include "common.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstring>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstdlib>
namespace pk {

int PromptCfg::resolve_index_or_throw(const std::string& target_lang) const {
    const std::string lang = target_lang.empty() ? default_lang : target_lang;
    int idx = lang_to_index(lang);
    if (idx < 0) {
        std::string sample;
        for (size_t i = 0; i < dict_keys.size() && i < 8; ++i)
            sample += (i ? ", " : "") + dict_keys[i];
        throw std::runtime_error("parakeet: unknown target_lang '" + lang +
                                 "'. Valid examples: " + sample + ", ...");
    }
    return idx;
}

static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : (uint32_t)gguf_get_val_u32(g,id);
}
static int32_t kv_i32(gguf_context* g, const char* k, int32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_i32(g,id);
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(a, a+n);
    }
    return out;
}
static std::vector<std::string> kv_str_arr(gguf_context* g, const char* k){
    std::vector<std::string> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_STRING){
        size_t n = gguf_get_arr_n(g,id);
        out.resize(n);
        for(size_t i=0;i<n;++i) out[i] = gguf_get_arr_str(g,id,i);
    }
    return out;
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int64_t id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}
ModelLoader::~ModelLoader(){
    // Free the weight buffer BEFORE the contexts. For the CPU path it is a
    // from_ptr buffer (free_buffer == NULL) that does NOT own its memory
    // (ctx_ does); for the device path it OWNS the device buffer. The device
    // upload path releases ctx_ as soon as the upload completes.
    if(repack_buf_) ggml_backend_buffer_free(repack_buf_);
    if(repack_ctx_) ggml_free(repack_ctx_);
    if(weights_buf_) ggml_backend_buffer_free(weights_buf_);
    if(device_ctx_) ggml_free(device_ctx_);
    if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_);
}
// Move eligible TQ2_0 weights into the CPU backend's CPU_REPACK buffer, which
// re-lays them out at upload for ggml's tiled TQ2_0 GEMM (block_tq2_0x4).
// PARAKEET_REPACK=0 keeps every weight in the plain zero-copy buffer.
bool ModelLoader::repack_ternary_weights(ggml_backend_t backend){
    const char* env = std::getenv("PARAKEET_REPACK");
    if (env && env[0] == '0') return false;
    if (!ggml_cpu_has_dotprod()) return false;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    auto get_extra = (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t))
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack_buft = nullptr;
    for (auto* b = get_extra ? get_extra(dev) : nullptr; b && *b; ++b) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) { repack_buft = *b; break; }
    }
    if (!repack_buft) return false;

    std::vector<std::string> names;
    for (auto& kv : tensors_) {
        const ggml_tensor* t = kv.second;
        if ((t->type == GGML_TYPE_TQ2_0 || t->type == GGML_TYPE_TQ1_0) &&
            ggml_n_dims(t) == 2 && t->ne[1] % 4 == 0 && t->ne[0] % 256 == 0)
            names.push_back(kv.first);
    }
    if (names.empty()) return false;

    struct ggml_init_params ip = { ggml_tensor_overhead() * (names.size() + 8), nullptr, true };
    repack_ctx_ = ggml_init(ip);
    std::vector<std::pair<ggml_tensor*, const void*>> ups;
    for (const auto& n : names) {
        ggml_tensor* s = tensors_[n];
        // TQ1_0 (1.69 bpw on disk) runs as TQ2_0: same scale + ternary codes, so the
        // conversion below is exact; the repack buffer then applies the GEMM layout.
        ggml_tensor* d = ggml_new_tensor(repack_ctx_, GGML_TYPE_TQ2_0, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, n.c_str());
        ups.emplace_back(d, s->data);
    }
    repack_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(repack_ctx_, repack_buft);
    if (!repack_buf_) { PK_LOG("repack: alloc failed, keeping plain weights"); return false; }
    std::vector<float>   row_f32;
    std::vector<uint8_t> tq2;
    for (auto& pr : ups) {
        const ggml_tensor* src = tensors_[ggml_get_name(pr.first)];
        if (src->type == GGML_TYPE_TQ2_0) {
            ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
            continue;
        }
        const int64_t K = src->ne[0], R = ggml_nrows(src);
        const auto* tq1 = ggml_get_type_traits(GGML_TYPE_TQ1_0);
        row_f32.resize((size_t) K);
        tq2.resize(ggml_nbytes(pr.first));
        for (int64_t r = 0; r < R; ++r) {
            tq1->to_float((const char*) pr.second + r * src->nb[1], row_f32.data(), K);
            ggml_quantize_chunk(GGML_TYPE_TQ2_0, row_f32.data(), tq2.data() + r * ggml_row_size(GGML_TYPE_TQ2_0, K), 0, 1, K, nullptr);
        }
        ggml_backend_tensor_set(pr.first, tq2.data(), 0, tq2.size());
    }
    for (auto& pr : ups) tensors_[ggml_get_name(pr.first)] = pr.first;
    return true;
}
bool ModelLoader::realize_weights(ggml_backend_t backend){
    if(weights_buf_) return true;                       // idempotent
    if(!backend || !ctx_){ PK_LOG("realize_weights: null backend/ctx"); return false; }

    if (ggml_backend_is_cpu(backend)) {
        // Fast path: borrow the host ctx memory directly (no copy).
        // The GGUF is loaded with no_alloc=false, so every tensor's data lives
        // in one contiguous ctx mem_buffer. Wrap that exact memory as a CPU
        // backend buffer (zero-copy: ggml_backend_cpu_buffer_from_ptr borrows
        // the ptr) and point every tensor's ->buffer at it, so graphs can
        // reference the loader tensors DIRECTLY as leaves (the gallocr treats
        // data!=NULL tensors as already-allocated and never copies them;
        // reshapes/views resolve at build time). Eliminates per-call recopy.
        // With repacked ternary weights the loader copy would be a second ~150 MB
        // of dead weights, so instead copy the remaining dense tensors (device
        // path below) and free the loader context.
        if (repack_ternary_weights(backend)) goto copy_rest;
        void*  base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if(!weights_buf_){ PK_LOG("realize_weights: buffer_from_ptr failed"); return false; }
        for(auto& kv : tensors_) kv.second->buffer = weights_buf_;
        return true;
    }
copy_rest:

    // Device path (CUDA/Metal/Vulkan/...): weights must live in a backend buffer.
    // ctx_ was created no_alloc=false (host-resident data), which
    // ggml_backend_alloc_ctx_tensors rejects (it asserts the ctx is no_alloc).
    // So mirror every weight into a no_alloc=true ctx, allocate THAT on the
    // backend, upload each tensor's bytes from the host source, and repoint the
    // name->tensor map at the device tensors. The host context is only a staging
    // allocation and can be released after all uploads complete.
    const size_t n = tensors_.size();
    struct ggml_init_params dp = {
        /*.mem_size  =*/ ggml_tensor_overhead() * (n + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    device_ctx_ = ggml_init(dp);
    if(!device_ctx_){ PK_LOG("realize_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> devmap; devmap.reserve(n);
    for (auto& kv : tensors_) {
        ggml_tensor* s = kv.second;
        if (s->buffer) { devmap.emplace(kv.first, s); continue; }   // already repacked
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);   // host source (valid in ctx_ mem buffer)
    }
    weights_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if(!weights_buf_){ PK_LOG("realize_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(devmap);   // graphs now reference the device-resident tensors

    // The device tensors and their backend buffer own everything needed for
    // inference. Keeping the no_alloc=false loader context alive retained a
    // second full copy of every model weight (notably ~708 MiB for the 0.6B Q5_K
    // model) for the lifetime of the process.
    ggml_free(ctx_);
    ctx_ = nullptr;
    return true;
}
bool ModelLoader::load(const std::string& path){
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ PK_LOG("gguf open failed: %s", path.c_str()); return false; }
    cfg_.arch        = kv_str(gguf_, "parakeet.arch");
    cfg_.feat_in     = kv_u32(gguf_, "parakeet.encoder.feat_in");
    cfg_.d_model     = kv_u32(gguf_, "parakeet.encoder.d_model");
    cfg_.n_layers    = kv_u32(gguf_, "parakeet.encoder.n_layers");
    cfg_.n_heads     = kv_u32(gguf_, "parakeet.encoder.n_heads");
    cfg_.ff_dim      = kv_u32(gguf_, "parakeet.encoder.ff_dim");
    cfg_.conv_kernel = kv_u32(gguf_, "parakeet.encoder.conv_kernel");
    cfg_.conv_norm_type = kv_str(gguf_, "parakeet.encoder.conv_norm_type", "batch_norm");
    cfg_.subsampling_factor = kv_u32(gguf_, "parakeet.encoder.subsampling_factor");
    cfg_.subsampling_conv_channels = kv_u32(gguf_, "parakeet.encoder.subsampling_conv_channels");
    cfg_.xscaling    = kv_bool(gguf_, "parakeet.encoder.xscaling", true);
    cfg_.pos_emb_max_len = kv_u32(gguf_, "parakeet.encoder.pos_emb_max_len", 5000);
    // cache-aware streaming / causal config (Phase 5). Absent for offline models
    // -> offline-safe defaults (regular style, no causal, streaming.present=false).
    cfg_.att_context_left  = kv_i32(gguf_, "parakeet.encoder.att_context_left", -1);
    cfg_.att_context_right = kv_i32(gguf_, "parakeet.encoder.att_context_right", -1);
    cfg_.att_context_style = kv_str(gguf_, "parakeet.encoder.att_context_style", "regular");
    cfg_.causal_downsampling = kv_bool(gguf_, "parakeet.encoder.causal_downsampling", false);
    cfg_.conv_causal = kv_bool(gguf_, "parakeet.encoder.conv_causal", false);
    // encoder.use_bias: false for nemotron (the attention/FFN linear projections
    // carry no bias tensor). Defaults true so existing models are unaffected.
    cfg_.use_bias = kv_bool(gguf_, "parakeet.encoder.use_bias", true);
    // Transformer encoder config (diarization models with RoPE attention).
    // Absent for ASR (FastConformer) models → safe defaults.
    cfg_.self_attention_model = kv_str(gguf_, "parakeet.encoder.self_attention_model", "");
    cfg_.qkv_bias = kv_bool(gguf_, "parakeet.encoder.qkv_bias", false);
    cfg_.pre_block_norm = kv_bool(gguf_, "parakeet.encoder.pre_block_norm", true);
    cfg_.rope_base = kv_f32(gguf_, "parakeet.encoder.rope_base", 10000.0f);
    cfg_.rotary_fraction = kv_f32(gguf_, "parakeet.encoder.rotary_fraction", 1.0f);
    // Prompt conditioning (multilingual nemotron). Orthogonal capability flag;
    // absent -> present=false and the engine skips the prompt stage entirely.
    cfg_.prompt.present = kv_bool(gguf_, "parakeet.prompt.present", false);
    if(cfg_.prompt.present){
        cfg_.prompt.num_prompts  = kv_u32(gguf_, "parakeet.prompt.num_prompts", 0);
        cfg_.prompt.default_lang = kv_str(gguf_, "parakeet.prompt.default_lang", "");
        cfg_.prompt.dict_keys = kv_str_arr(gguf_, "parakeet.prompt.dictionary.keys");
        cfg_.prompt.dict_vals = kv_i32_arr(gguf_, "parakeet.prompt.dictionary.values");
    }
    if(cfg_.att_context_style != "regular"){
        StreamingCfg& s = cfg_.streaming;
        s.chunk_size = kv_i32_arr(gguf_, "parakeet.streaming.chunk_size");
        s.shift_size = kv_i32_arr(gguf_, "parakeet.streaming.shift_size");
        s.pre_encode_cache_size = kv_i32_arr(gguf_, "parakeet.streaming.pre_encode_cache_size");
        s.cache_drop_size = kv_i32(gguf_, "parakeet.streaming.cache_drop_size", 0);
        s.last_channel_cache_size = kv_i32(gguf_, "parakeet.streaming.last_channel_cache_size", 0);
        s.valid_out_len = kv_i32(gguf_, "parakeet.streaming.valid_out_len", 0);
        s.drop_extra_pre_encoded = kv_i32(gguf_, "parakeet.streaming.drop_extra_pre_encoded", 0);
        s.present = true;
    }
    cfg_.sample_rate = kv_u32(gguf_, "parakeet.preprocessor.sample_rate", 16000);
    cfg_.n_mels      = kv_u32(gguf_, "parakeet.preprocessor.n_mels");
    cfg_.n_fft       = kv_u32(gguf_, "parakeet.preprocessor.n_fft");
    cfg_.win_length  = kv_u32(gguf_, "parakeet.preprocessor.win_length");
    cfg_.hop_length  = kv_u32(gguf_, "parakeet.preprocessor.hop_length");
    cfg_.preemph     = kv_f32(gguf_, "parakeet.preprocessor.preemph", 0.0f);
    cfg_.mag_power   = kv_f32(gguf_, "parakeet.preprocessor.mag_power", 2.0f);
    cfg_.normalize   = kv_str(gguf_, "parakeet.preprocessor.normalize", "per_feature");
    cfg_.log_zero_guard = kv_f32(gguf_, "parakeet.preprocessor.log_zero_guard", 0.0f);
    cfg_.pred_hidden = kv_u32(gguf_, "parakeet.decoder.pred_hidden");
    cfg_.pred_rnn_layers = kv_u32(gguf_, "parakeet.decoder.pred_rnn_layers");
    cfg_.joint_hidden = kv_u32(gguf_, "parakeet.joint.joint_hidden");
    cfg_.joint_activation = kv_str(gguf_, "parakeet.joint.activation");
    cfg_.max_symbols = kv_u32(gguf_, "parakeet.decoding.max_symbols", 10);
    cfg_.vocab_size  = kv_u32(gguf_, "parakeet.vocab_size");
    cfg_.blank_id    = kv_u32(gguf_, "parakeet.blank_id");
    // diarization config (absent for ASR models → present=false)
    if (gguf_find_key(gguf_, "parakeet.diar.n_speakers") >= 0) {
        auto& d = cfg_.diarization;
        d.present = true;
        d.n_speakers = kv_u32(gguf_, "parakeet.diar.n_speakers");
        d.tf_d_model = kv_u32(gguf_, "parakeet.diar.tf_d_model");
        d.upsample_factor = kv_u32(gguf_, "parakeet.diar.upsample_factor");
        d.frame_resolution_sec = kv_f32(gguf_, "parakeet.diar.frame_resolution_sec", 0.01f);
        d.onset_threshold = kv_f32(gguf_, "parakeet.diar.onset_threshold", 0.5f);
        d.offset_threshold = kv_f32(gguf_, "parakeet.diar.offset_threshold", 0.5f);
        // Streaming (speaker cache) config, in ENCODER frames as in NeMo
        // SortformerModules. Defaults are the Nemotron-3-Diarization values,
        // for GGUFs converted before these keys were written.
        d.streaming_mode         = kv_bool(gguf_, "parakeet.diar.streaming_mode", true);
        d.chunk_len              = (int32_t)kv_u32(gguf_, "parakeet.diar.chunk_len", 264);
        d.spkcache_len           = (int32_t)kv_u32(gguf_, "parakeet.diar.spkcache_len", 264);
        d.fifo_len               = (int32_t)kv_u32(gguf_, "parakeet.diar.fifo_len", 0);
        d.spkcache_update_period = (int32_t)kv_u32(gguf_, "parakeet.diar.spkcache_update_period", 264);
        d.spkcache_sil_frames_per_spk = (int32_t)kv_u32(gguf_, "parakeet.diar.spkcache_sil_frames_per_spk", 1);
        d.sil_threshold          = kv_f32(gguf_, "parakeet.diar.sil_threshold", 0.2f);
        d.pred_score_threshold   = kv_f32(gguf_, "parakeet.diar.pred_score_threshold", 0.25f);
        d.scores_boost_latest    = kv_f32(gguf_, "parakeet.diar.scores_boost_latest", 0.05f);
        d.strong_boost_rate      = kv_f32(gguf_, "parakeet.diar.strong_boost_rate", 0.75f);
        d.weak_boost_rate        = kv_f32(gguf_, "parakeet.diar.weak_boost_rate", 1.5f);
        d.min_pos_scores_rate    = kv_f32(gguf_, "parakeet.diar.min_pos_scores_rate", 0.5f);
        d.use_learnable_sil_emb  = kv_bool(gguf_, "parakeet.diar.use_learnable_sil_emb",
            gguf_find_tensor(gguf_, "sortformer_modules.learnable_sil_emb") >= 0);
    }
    // durations array (stored as INT32 by the converter)
    { int64_t id = gguf_find_key(gguf_, "parakeet.tdt.durations");
      if(id>=0 && gguf_get_arr_type(gguf_,id)==GGUF_TYPE_INT32){
          size_t n = gguf_get_arr_n(gguf_,id);
          const int32_t* a = (const int32_t*)gguf_get_arr_data(gguf_,id);
          cfg_.tdt_durations.assign(a, a+n); } }
    // tokenizer pieces STRING array
    { int64_t id = gguf_find_key(gguf_, "parakeet.tokenizer.pieces");
      if(id>=0 && gguf_get_arr_type(gguf_,id)==GGUF_TYPE_STRING){
          size_t n = gguf_get_arr_n(gguf_,id);
          cfg_.tokenizer_pieces.resize(n);
          for(size_t i=0;i<n;++i)
              cfg_.tokenizer_pieces[i] = gguf_get_arr_str(gguf_,id,i); } }
    // tensors
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for(int64_t i=0;i<nt;++i){ const char* nm = gguf_get_tensor_name(gguf_,i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm); if(t) tensors_[nm]=t; }
    return cfg_.d_model>0 && (cfg_.vocab_size>0 || cfg_.arch=="diarization");
}
ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it==tensors_.end()? nullptr : it->second;
}
}
