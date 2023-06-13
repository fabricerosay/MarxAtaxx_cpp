#include <iostream>
#include <fstream>
#include <array>
#include <chrono>
#include <random>
#include <memory>
#include <sstream>
#include <cstring>
#include <string>
#include <thread>
#include <mutex>

#include <immintrin.h>

#include "assert.h"

struct ModelWeights
{
    int16_t W0[62985600];
    float W1[128 * 32];
    float B1[32];
    float W2[32 * 32];
    float B2[32];
    float W3[32];
    float B3[1];
    void load_weights(std::string w);

    ModelWeights(){};
    ModelWeights(std::string w) { load_weights(w); }
};

const std::string WeightsPath="path to weights here";
  
void ModelWeights::load_weights(std::string w)
{
    int cpt = 0;

    std::ifstream w1(WeightsPath + w + "10weights.dat", std::ios::binary);
    while (w1.read(reinterpret_cast<char *>(&W1[cpt]), sizeof(float)))
    {
        cpt++;
    };

    cpt = 0;
    std::ifstream b1(WeightsPath + w + "10bias.dat", std::ios::binary);
    while (b1.read(reinterpret_cast<char *>(&B1[cpt]), sizeof(float)))
    {
        cpt++;
    };
    cpt = 0;
    std::ifstream w2(WeightsPath+ w + "20weights.dat", std::ios::binary);
    while (w2.read(reinterpret_cast<char *>(&W2[cpt]), sizeof(float)))
    {

        cpt++;
    };

    cpt = 0;
    std::ifstream b2(WeightsPath + w + "20bias.dat", std::ios::binary);
    while (b2.read(reinterpret_cast<char *>(&B2[cpt]), sizeof(float)))
    {
        cpt++;
    };
    cpt = 0;
    std::ifstream w3(WeightsPath + w + "30weights.dat", std::ios::binary);
    while (w3.read(reinterpret_cast<char *>(&W3[cpt]), sizeof(float)))
    {
        cpt++;
    };
    cpt = 0;

    std::ifstream b3(WeightsPath + w + "30bias.dat", std::ios::binary);
    while (b3.read(reinterpret_cast<char *>(&B3[cpt]), sizeof(float)))
    {
        cpt++;
    };
    cpt = 0;
    std::ifstream w0(WeightsPath + w + "00.dat", std::ios::binary);
    while (w0.read(reinterpret_cast<char *>(&W0[cpt]), sizeof(int16_t)))
    {
        cpt++;
    };
};

struct Evaluator
{
    Evaluator() { weights = new ModelWeights("hW"); };
    Evaluator(const ModelWeights *weights) : weights(weights){};
    ~Evaluator(){};
    int32_t apply();
    const ModelWeights *weights;
    int active_features[25] = {0};
    float accumulator_raw[128] = {0};
    float accumulator_clamped[128] = {0};
    float out_raw[32] = {0};
    float out_clamped[32] = {0};
};

void nn_compute_layer(const float *B, const float *I, const float *W, float *O, int idim, int odim);

void clamp0(const float *I, float *out, const int idim)
{
    constexpr int register_width = 256 / 32;
    int num_chunks = idim / register_width;
    __m256 regs;
    __m256 azeros = _mm256_setzero_ps();
    // std::memcpy(new_acc, W + 128 * active_features[0], 256);
    for (int i = 0; i < num_chunks; ++i)
    {
        // Now we do 1 memory operation instead of 2 per loop iteration.
        regs = _mm256_max_ps(_mm256_loadu_ps(&I[i * register_width]), azeros);
        _mm256_storeu_ps(&out[i * register_width], regs);
    }
}

void nn_compute_layer(const float *B, const float *I, const float *W, float *O, int idim, int odim)
{
    for (int o = 0; o < odim; o++)
    {
        float sum = B[o];

        const int offset = o * idim;

        __m256 dot0 = _mm256_mul_ps(_mm256_loadu_ps(&I[0]), _mm256_loadu_ps(&W[offset + 0]));
        __m256 dot1 = _mm256_mul_ps(_mm256_loadu_ps(&I[8]), _mm256_loadu_ps(&W[offset + 8]));
        __m256 dot2 = _mm256_mul_ps(_mm256_loadu_ps(&I[16]), _mm256_loadu_ps(&W[offset + 16]));
        __m256 dot3 = _mm256_mul_ps(_mm256_loadu_ps(&I[24]), _mm256_loadu_ps(&W[offset + 24]));

        for (int i = 32; i < idim; i += 32)
        {
            dot0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(&I[i + 0]), _mm256_loadu_ps(&W[offset + i + 0])), dot0);
            dot1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(&I[i + 8]), _mm256_loadu_ps(&W[offset + i + 8])), dot1);
            dot2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(&I[i + 16]), _mm256_loadu_ps(&W[offset + i + 16])), dot2);
            dot3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(&I[i + 24]), _mm256_loadu_ps(&W[offset + i + 24])), dot3);
        }

        const __m256 dot01 = _mm256_add_ps(dot0, dot1);
        const __m256 dot23 = _mm256_add_ps(dot2, dot3);
        const __m256 dot03 = _mm256_add_ps(dot01, dot23);
        const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(dot03), _mm256_extractf128_ps(dot03, 1));
        const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
        const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));

        sum += _mm_cvtss_f32(r1);
        O[o] = sum;
    }
}

void refresh_accumulator(const int16_t *W, float *new_acc, const int *active_features)
{
    // The compiler should use one register per value, and hopefully
    // won't spill anything. Always check the assembly generated to be sure!
    constexpr int register_width = 256 / 16;
    constexpr int num_chunks = 128 / register_width;
    __m256i regs[num_chunks] = {_mm256_setzero_si256()};
    // std::memcpy(new_acc, W + 128 * active_features[0], 256);
    int a;
    for (int k = 0; k < 25; k++)
    {
        a = active_features[k];
        for (int i = 0; i < num_chunks; ++i)
        {

            // Now we do 1 memory operation instead of 2 per loop iteration.
            regs[i] = _mm256_add_epi16(regs[i], _mm256_loadu_si256((__m256i *)&W[a * 128 + i * register_width]));
        }
    }

    // Only after all the accumulation is done do the write.
    for (int i = 0; i < num_chunks; ++i)
    {
        _mm256_storeu_ps(&new_acc[i * register_width], _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(regs[i], 0))));
        _mm256_storeu_ps(&new_acc[i * register_width + 8], _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(regs[i], 1))));
    }
}

int32_t Evaluator::apply()
{

    refresh_accumulator(weights->W0, accumulator_raw, active_features);

    clamp0(accumulator_raw, accumulator_clamped, 128);

    nn_compute_layer(weights->B1, accumulator_clamped, weights->W1, out_raw, 128, 32);

    clamp0(out_raw, out_clamped, 32);

    nn_compute_layer(weights->B2, out_clamped, weights->W2, out_raw, 32, 32);

    clamp0(out_raw, out_clamped, 32);

    float eval = (weights->B3)[0];

    for (int i = 0; i < 32; i++)
    {

        eval += (weights->W3)[i] * out_clamped[i];
    }

    return eval;
}

using namespace std;

namespace board
{

    static constexpr int N = 7;
    static constexpr int NN = N * N;

#define SquareOf(X) __builtin_ctzll(X)
#define Bitloop(X) for (; X; X &= X - 1)
#define BB uint64_t
#define u8 uint8_t
    enum Color
    {
        BLACK,
        WHITE,
        DRAW,
        NONE
    };

    [[nodiscard]] constexpr int count(BB data)
    {
        return __builtin_popcountll(data);
    };
    [[nodiscard]] constexpr void set(BB &data, int sq)
    {
        data |= (1ULL << sq);
    };
    [[nodiscard]] constexpr bool get(BB data, int sq)
    {
        return (data >> sq) & 1ULL;
    };

    [[nodiscard]] constexpr BB singlesBB(BB data)
    {
        return ((data << 1 | data << 9 | data >> 7 | data << 8 | data >> 8 | data >> 1 | data >> 9 | data << 7) & 0x7f7f7f7f7f7f7fULL);
    };

    [[nodiscard]] constexpr BB doublesBB(BB data)
    {
        return (((data << 2 | data << 10 | data << 18 | data >> 6 | data >> 14 | data << 17 | data >> 15) & 0x7e7e7e7e7e7e7eULL) |
                ((data << 16 | data >> 16) & 0x7f7f7f7f7f7f7fULL) |
                ((data >> 2 | data >> 10 | data >> 18 | data << 6 | data << 14 | data << 15 | data >> 17) & 0x3f3f3f3f3f3f3fULL));
    };
    [[nodiscard]] constexpr bool is_empty(BB data)
    {
        return data == 0;
    };
    [[nodiscard]] constexpr bool is_full(BB data)
    {
        return data == 0x7f7f7f7f7f7f7fULL;
    };
    static constexpr uint64_t FULL = 0x7f7f7f7f7f7f7fULL;

    void print(BB data)
    {
        std::cout << "\n";
        for (int i = 0; i < 55; i++)
        {
            if (i % 8 == 7)
            {
                std::cout << "\n";
            }
            else
            {
                if (get(data, i))
                {
                    std::cout << "x";
                }
                else
                {
                    std::cout << "_";
                }
            }
        }
    };

    [[nodiscard]] constexpr BB singles(int sq)
    {
        BB b = 1ULL << sq;
        return singlesBB(b);
    };
    [[nodiscard]] constexpr BB doubles(int sq)
    {
        BB b = 1ULL << sq;
        return doublesBB(b);
    };

#define Singles(X) singles(X)

    static const constexpr array<BB, 64> SINGLES = {Singles(0), Singles(1), Singles(2), Singles(3), Singles(4), Singles(5), Singles(6), Singles(7),
                                                    Singles(8), Singles(9), Singles(10), Singles(11), Singles(12), Singles(13), Singles(14), Singles(15),
                                                    Singles(16), Singles(17), Singles(18), Singles(19), Singles(20), Singles(21), Singles(22), Singles(23),
                                                    Singles(24), Singles(25), Singles(26), Singles(27), Singles(28), Singles(29), Singles(30), Singles(31),
                                                    Singles(32), Singles(33), Singles(34), Singles(35), Singles(36), Singles(37), Singles(38), Singles(39),
                                                    Singles(40), Singles(41), Singles(42), Singles(43), Singles(44), Singles(45), Singles(46), Singles(47),
                                                    Singles(48), Singles(49), Singles(50), Singles(51), Singles(52), Singles(53), Singles(54), Singles(55),
                                                    Singles(56), Singles(57), Singles(58), Singles(59), Singles(60), Singles(61), Singles(62), Singles(63)};

#define Doubles(X) doubles(X)

    static const constexpr array<BB, 64> DOUBLES = {Doubles(0), Doubles(1), Doubles(2), Doubles(3), Doubles(4), Doubles(5), Doubles(6), Doubles(7),
                                                    Doubles(8), Doubles(9), Doubles(10), Doubles(11), Doubles(12), Doubles(13), Doubles(14), Doubles(15),
                                                    Doubles(16), Doubles(17), Doubles(18), Doubles(19), Doubles(20), Doubles(21), Doubles(22), Doubles(23),
                                                    Doubles(24), Doubles(25), Doubles(26), Doubles(27), Doubles(28), Doubles(29), Doubles(30), Doubles(31),
                                                    Doubles(32), Doubles(33), Doubles(34), Doubles(35), Doubles(36), Doubles(37), Doubles(38), Doubles(39),
                                                    Doubles(40), Doubles(41), Doubles(42), Doubles(43), Doubles(44), Doubles(45), Doubles(46), Doubles(47),
                                                    Doubles(48), Doubles(49), Doubles(50), Doubles(51), Doubles(52), Doubles(53), Doubles(54), Doubles(55),
                                                    Doubles(56), Doubles(57), Doubles(58), Doubles(59), Doubles(60), Doubles(61), Doubles(62), Doubles(63)};

    static constexpr array<array<BB, 64>, 2> ZOBRIST{{{0xdf32a11764efb447ULL, 0x5f64832ee328d3d5ULL, 0x67186ab4775e587fULL, 0x6a478254df2e8a16ULL, 0xeb914d1e9018ecfbULL, 0x88fb46b83be037ccULL, 0xd59447789724dfb8ULL, 0x96255ff245389e7eULL,
                                                       0x7ae2a111addff176ULL, 0xcbe2770cb96c1ec8ULL, 0xc82e7c8700fdeeddULL, 0xe65fd9aa519616eULL, 0xdbce94ee5bb4647aULL, 0xbb1088cbe7771d8cULL, 0x32e32f15cc7af3f3ULL, 0x39d1ccfd753d8d85ULL,
                                                       0x1c9bf13416822904ULL, 0x5b2a467bb543e9f9ULL, 0x7e550dd2f524392cULL, 0x4549c73ab0987f83ULL, 0xe2dcbd51d4a8e37eULL, 0x2dc7ab5f36185d7ULL, 0xeca2b747474b7a21ULL, 0x758b4078521124b2ULL,
                                                       0x580670c5a7634a89ULL, 0x7ab0dd692f9fe3bdULL, 0xe1ac61b7bb7ea83ULL, 0x38798d5d3ebbb115ULL, 0xeb5269528d51cfc2ULL, 0x9c7f7e40da007893ULL, 0xadd9829f15ad119eULL, 0x99422933d6aaee98ULL,
                                                       0xc51336420b2668eULL, 0x99be495928d502d4ULL, 0x1181212a60a3fd36ULL, 0xd16faaba34c1d44bULL, 0x480b0249c581a973ULL, 0xf62fb645cb1bb861ULL, 0x283ec6d9cb5a453eULL, 0xb624d64826a6ae39ULL,
                                                       0x971ac4942e317cbbULL, 0x1578e24fdafda327ULL, 0x3879e82ffec60cc2ULL, 0xb094cf75c7ce1c44ULL, 0x1139daa760dddff9ULL, 0xe55c1634a4ab64f4ULL, 0xd1de322b07bb6c39ULL, 0xc6589f2863fd3a11ULL,
                                                       0x80950f0a74a04fdfULL, 0x1b63fc9c21836f23ULL, 0x60ba706207168b74ULL, 0xae25b92f1056ebf1ULL, 0x2a353c944f01b050ULL, 0xff3896c2eb23d4f8ULL, 0xb612b64b99fb6d20ULL, 0xd15c8ca06c117c66ULL,
                                                       0x92a6851f2e50af6eULL, 0x7a76df23b7f51b77ULL, 0x8ddf77304deada57ULL, 0x7508ae07aa3b5709ULL, 0xd83a7264c5d862b5ULL, 0xcb741a14167fd11bULL, 0xd183993059f00e7eULL, 0x73dcf4a90baf5fb4ULL},
                                                      {0xdb5ac8a8c94be745ULL, 0x912f151b03b8b740ULL, 0x18dea02f7de687d8ULL, 0x94aac2903c2b31b8ULL, 0x66c955670e8f8670ULL, 0x1efdfc63d702512cULL, 0x192d03c0375aa845ULL, 0x77bd357bdd90e7d7ULL,
                                                       0xb34cd76e19726d91ULL, 0x6bbe0d1e46197975ULL, 0xc7cd94d44c6e5e1fULL, 0x78f2070195a995efULL, 0xf3004a9328c6b9d2ULL, 0x4e90fb8f0ebe164cULL, 0x593d28d82852a578ULL, 0x92d2416358db1a04ULL,
                                                       0x5e73aa9e6306e27aULL, 0x3342146c923da47dULL, 0x597fed519bebc8dfULL, 0xc912b4af3e522fbaULL, 0xae34545af63a1397ULL, 0x2e51c0f161f27a98ULL, 0xb49e2d84ffcd0cceULL, 0xff38244399b57422ULL,
                                                       0x71c862bead2384b4ULL, 0x6b90165dea9d7f99ULL, 0xb1c3f7d138fec59ULL, 0xa4e9175f9d55c3e8ULL, 0x83a26b54f53dbba1ULL, 0x39472a8816cba8e0ULL, 0x8f05384a95dfa7dbULL, 0xb611c9c16688633eULL,
                                                       0x8c5250dad19431e1ULL, 0x83895011d1613878ULL, 0x530222bb2d20a867ULL, 0xe99f85cc1968c2c7ULL, 0xe4912de01c01a3f0ULL, 0xd297d2cf510899aaULL, 0x607588afdd977820ULL, 0x38050446fc6fd62dULL,
                                                       0x5a7d273c56610797ULL, 0x13fc9844ad7c5fe9ULL, 0x1f0de65c1f0670bbULL, 0x691e4efaa7a119d0ULL, 0x9168b4526f82facaULL, 0xdb63baa16d2ecc5eULL, 0x27d28c18d9ee77d2ULL, 0xb6d69d352f0f38d6ULL,
                                                       0x153115867d64980eULL, 0xe3f19475b1b7b395ULL, 0x99dbf196b456014bULL, 0x1370110135c5d403ULL, 0x4c06bfaae5d80bd9ULL, 0xab7bf59bf468f678ULL, 0x7138d4f111d00645ULL, 0xd01413d20a163a08ULL,
                                                       0x3beab9d17d99bd54ULL, 0xc26c1221c2cef149ULL, 0x5d6a9f84744e350bULL, 0xdc51d42fee5fdebbULL, 0xce0fac0443a2b7cbULL, 0xc4f22a9d204bb05ULL, 0xeabd0f057eb99c4cULL, 0x4f3855db93cc271bULL}}};
    // {0x3b84a2bc1c00f5efULL, 0x540591ac50deb1a7ULL, 0x9e5438a02f559ca3ULL, 0x7063a48359c124f3ULL, 0x51fe9bf820661d92ULL, 0x5dc230714178151dULL, 0x6d8004c72f1e5a42ULL, 0x1d1c990dd1353df9ULL,
    //          0xa9bc80734693ba62ULL, 0x7ab76b1b828b480dULL, 0x92be9d8d7bb517a5ULL, 0x3ea054fd3cac2f95ULL, 0x79660ed04850d1ffULL, 0x177b315383f518caULL, 0x3636fd500f66bbdbULL, 0xe425cce97b96d5d8ULL,
    //          0x49119c79a7227799ULL, 0x56ac3746900e7c62ULL, 0xde6cdc21fe7befbbULL, 0x8aedc6cfba61b0dbULL, 0x9f4009bd328d805cULL, 0x805f7f170c15ae08ULL, 0x33c0077f8b637a00ULL, 0x1447b881f73acc56ULL,
    //          0xb0acc713beeffa64ULL, 0x9704d95e7066b7beULL, 0xef1d616b54f7c9c9ULL, 0x8593fcc4fe1ffdc6ULL, 0x1704cddebc06dba2ULL, 0x89d22ad0286a8a1dULL, 0xdb9b235f5a7761d4ULL, 0x4024c6aadef0ecb1ULL,
    //          0x9e393045e1c97a73ULL, 0x26c11d94a5026d27ULL, 0x62ac950c797bb1eeULL, 0x1be8c10503b00829ULL, 0xaae6dbdf35e5430aULL, 0x1dd8d1d83bcbef9eULL, 0xe2ee6d09ac795460ULL, 0x8dc4e32adadc7d8bULL,
    //          0x24075ba3a9e17c7aULL, 0x9668d94ecc94abbdULL, 0xfa2c1384b6baa543ULL, 0x51c904fe21801171ULL, 0x6e9fd48c6f866de3ULL, 0x103d051cef11d515ULL, 0x65dae4dae1ae18ccULL, 0xea62656bafeb8878ULL,
    //          0xe3983322cfdb6c1aULL, 0x60c94215663559ceULL, 0x462fa0fe1c856599ULL, 0x4bd15c87b12df83ULL, 0xe7b2d479d57acea2ULL, 0x20cabd0f8990f47ULL, 0x5f77f65d2b1e740bULL, 0x32f944aa97925c43ULL,
    //          0xa2b2f2cce638d5d8ULL, 0x432bbe368216149bULL, 0x541b911119d1a9b9ULL, 0x967644c18834af2dULL, 0xeaf7f7dd572693bfULL, 0xd71a01ed601fb9c2ULL, 0xb6638eed56a1aa51ULL, 0xa16bf6605c6793d6ULL},
    //     }};

    static constexpr BB TURN = 0xfae9e1a62375b174ULL;

#define Board std::array<BB, 2>

    struct Move
    {
    public:
        Move() = default;
        constexpr explicit Move(const int8_t f, const int8_t t) : sqfrom(f), sqto(t){};
        [[nodiscard]] constexpr int from() const noexcept
        {
            return sqfrom;
        };
        [[nodiscard]] constexpr int to() const noexcept
        {
            return sqto;
        };
        void print()
        {
            std::cout << "Move(" << static_cast<int>(sqfrom) << "," << static_cast<int>(sqto) << ")";
        };

    private:
        int8_t sqfrom;
        int8_t sqto;
    };
    std::ostream &operator<<(std::ostream &os, const Move &move)
    {
        static const std::string coords[64] = {
            "g1", "f1", "e1", "d1", "c1", "b1", "a1", "",
            "g2", "f2", "e2", "d2", "c2", "b2", "a2", "",
            "g3", "f3", "e3", "d3", "c3", "b3", "a3", "",
            "g4", "f4", "e4", "d4", "c4", "b4", "a4", "",
            "g5", "f5", "e5", "d5", "c5", "b5", "a5", "",
            "g6", "f6", "e6", "d6", "c6", "b6", "a6", "",
            "g7", "f7", "e7", "d7", "c7", "b7", "a7", "",
            "g7", "f7", "e7", "d7", "c7", "b7", "a7", ""};

        if (move.to() == -1)
        {
            os << "0000";
            return os;
        }
        if (move.to() == move.from())
        {
            // int ind = move.to() % 8 + 7 * (move.to() >> 3);
            os << coords[move.to()];
            return os;
        }
        // int indt = move.to() % 8 + 7 * (move.to() >> 3);
        // int indf = move.to() % 8 + 7 * (move.from() >> 3);
        os << coords[move.from()] << coords[move.to()];
        return os;
    }

    static const constexpr Move NONEMOVE = Move(-2, -2);
    static const constexpr Move PASSMOVE = Move(-1, -1);

    struct Result
    {
        Result(bool o, int8_t w) : over(o), winner(w){};
        bool over;
        int winner;
    };

    struct Undo
    {
        BB u;
        uint8_t fifty;
    };
    struct Game
    {
        Game() noexcept = default;
        Game(bool player) : board(Board{0b0000000001000000000000000000000000000000000000000000000000000001ULL,
                                        0b0000000000000001000000000000000000000000000000000000000001000000ULL}),
                            player(player), round(0), fifty(0), hash(ZOBRIST[0][0] ^ ZOBRIST[0][54] ^ ZOBRIST[1][6] ^ ZOBRIST[1][48]){};

        [[nodiscard]] Board get_board()
        {
            return board;
        };
        [[nodiscard]] bool get_player()
        {
            return player;
        };
        [[nodiscard]] int16_t get_round()
        {
            return round;
        };
        [[nodiscard]] uint64_t get_hash()
        {
            return hash;
        };
        [[nodiscard]] uint8_t get_fifty()
        {
            return fifty;
        };
        [[nodiscard]] constexpr BB us() const noexcept
        {
            return board[static_cast<int>(player)];
        };
        [[nodiscard]] constexpr BB other() const noexcept
        {
            return board[static_cast<int>(!player)];
        };
        [[nodiscard]] constexpr BB both() const noexcept
        {
            return (board[0] | board[1]);
        };
        [[nodiscard]] constexpr BB empties() const noexcept
        {
            return ~both() & FULL;
        }
        [[nodiscard]] constexpr bool is_full() const noexcept
        {
            return both() == FULL;
        }

        [[nodiscard]] constexpr bool isOver() const noexcept
        {
            const bool over = !us() || !other() || is_full() || (fifty >= 100);
            return over;
        };

        void gen_key()
        {
            hash = 0ULL;
            BB black = board[0];
            int sq;
            Bitloop(black)
            {
                sq = SquareOf(black);
                hash ^= ZOBRIST[0][sq];
            }
            black = board[1];
            Bitloop(black)
            {
                sq = SquareOf(black);
                hash ^= ZOBRIST[1][sq];
            }
            if (player)
                hash ^= TURN;
        }

        constexpr Undo play(const Move move)
        {
            const int sqf = move.from();
            round += 1;
            auto ofifty = fifty;
            fifty += 1;
            if (sqf == -1)
            {
                player = !player;

                hash ^= TURN;

                return {0ULL, ofifty};
            }
            const BB sqt = 1ULL << move.to();
            const BB turned = SINGLES[move.to()] & board[!player];

            // place stone
            board[player] ^= (1ULL << sqf) | sqt;
            // capture
            board[player] ^= turned;
            board[!player] ^= turned;

            // update hash

            if (sqf != move.to())
            {
                hash ^= ZOBRIST[static_cast<int>(player)][sqf];
                fifty = 0;
            }
            hash ^= ZOBRIST[static_cast<int>(player)][move.to()];
            BB turned_loop = turned;
            Bitloop(turned_loop)
            {
                int sq = SquareOf(turned_loop);
                hash ^= ZOBRIST[static_cast<int>(player)][sq];
                hash ^= ZOBRIST[static_cast<int>(!player)][sq];
            }
            hash ^= TURN;
            player = !player;
            return {turned, ofifty};
        };

        constexpr void undo(const Move move, Undo inf)
        {
            const int sqf = move.from();
            round -= 1;
            fifty = inf.fifty;
            if (sqf == -1)
            {
                player = !player;
                hash ^= TURN;
                return;
            }
            BB turned = inf.u;
            const BB sqt = 1ULL << move.to();
            board[!player] ^= (1ULL << sqf) | sqt;
            board[!player] ^= turned;
            board[player] ^= turned;

            player = !player;
            if (sqf != move.to()) // peut faire plus vite avec zobr(sqf)|zobr(sqt)
            {
                hash ^= ZOBRIST[static_cast<int>(player)][sqf];
            }
            hash ^= ZOBRIST[static_cast<int>(player)][move.to()];
            Bitloop(turned)
            {
                int sq = SquareOf(turned);
                hash ^= ZOBRIST[static_cast<int>(player)][sq];
                hash ^= ZOBRIST[static_cast<int>(!player)][sq];
            }
            hash ^= TURN;
            return;
        };

        int gen_moves(Move *moves, int *scores, Move hmove = NONEMOVE, bool sort = true)
        {
            static const std::array<int, 9> jumping_penalties = {0, 0, 0, 100, 200, 200, 300, 300, 400};
            static const std::array<std::array<int, 9>, 2> score_by_captures{{
                {200, 300, 400, 600, 700, 800, 900, 1000, 1100}, // Single moves
                {0, 100, 200, 300, 400, 500, 600, 700, 800}      // Double moves
            }};

            // if (isOver())
            // {
            //     return 0;
            // }
            int cpt = 0;
            int endsort = 0;
            int score;
            int8_t sq;
            if (hmove.from() != -2)
            {
                moves[0] = hmove;
                cpt = 1;
                endsort = 1;
            }
            BB test = singlesBB(us()) & (empties());
            Bitloop(test)
            {
                sq = static_cast<int8_t>(SquareOf(test));
                if (sq != hmove.from())
                {
                    moves[cpt] = Move(sq, sq);

                    scores[cpt] = score_by_captures[0][board::count(singles(sq) & other())];
                    score = scores[cpt];
                    int k = cpt - 1;
                    while (k >= endsort)
                    {
                        if (scores[k] < score)
                        {
                            swap(scores[k + 1], scores[k]);
                            swap(moves[k + 1], moves[k]);
                            k -= 1;
                        }
                        else
                        {
                            break;
                        }
                    }
                    cpt += 1;
                }
            }
            int64_t bp = us();
            Bitloop(bp)
            {
                int sqf(SquareOf(bp));
                uint64_t test = DOUBLES[sqf] & (empties());
                Bitloop(test)
                {
                    int8_t sqt(SquareOf(test));
                    if (sqf != hmove.from() || sqt != hmove.to())
                    {
                        moves[cpt] = Move(sqf, sqt);
                        // score = board::count(singles(sqt) & other()) - board::count(singles(sqf) & us());
                        scores[cpt] = score_by_captures[1][board::count(singles(sqt) & other())] - jumping_penalties[board::count(singles(sqf) & us())];
                        ;
                        ;
                        score = scores[cpt];
                        int k = cpt - 1;
                        while (k >= endsort)
                        {
                            if (scores[k] < score)
                            {
                                swap(scores[k + 1], scores[k]);
                                swap(moves[k + 1], moves[k]);
                                k -= 1;
                            }
                            else
                            {
                                break;
                            }
                        }
                        cpt += 1;
                    }
                }
            }
            if (cpt == 0)
            {
                moves[cpt] = PASSMOVE;
                cpt = 1;
            }
            return cpt;
        };
        [[nodiscard]] constexpr int count_moves() const noexcept
        {
            if (isOver())
            {
                return 0;
            }
            const BB test = singlesBB(us()) & (empties());
            int nmoves = count(test);
            BB bp = us();
            Bitloop(bp)
            {
                const int sqf = SquareOf(bp);
                const BB test = DOUBLES[sqf] & (empties());
                nmoves += count(test);
            }
            if (nmoves == 0)
            {
                nmoves = 1;
            }
            return nmoves;
        };
        constexpr bool legal_move(Move move)
        {
            return get(empties(), move.to()) && (((SINGLES[move.to()] & us()) && (move.to() == move.from())) || ((DOUBLES[move.to()] && us()) && (move.to() != move.from())));
        }
        void from_fen(const string &fen)
        {
            board[0] = 0ULL;
            board[1] = 0ULL;
            std::string boardString;
            char turnChar;
            std::string fiftyMovesString;
            int line = 0;
            int column = 0;
            int cpt = 54 - 8 * line - column;
            std::stringstream ss(fen);

            if (!(ss >> boardString))
            {
                std::cout << "[-] Fen string is blank" << std::endl;
                exit(0);
            }

            if (!(ss >> turnChar))
            {
                std::cout << "[-] No turn found on fen: " << fen << std::endl;
                exit(0);
            }

            if (ss >> fiftyMovesString)
                fifty = std::stoi(fiftyMovesString);

            for (char c : boardString)
            {
                switch (c)
                {
                case '/':
                    line += 1;
                    column = 0;
                    break;
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                    column += c - '0';
                    break;
                case 'x':
                case 'X':
                case 'b':
                case 'B':
                    board[0] |= 1ULL << (54 - 8 * line - column);
                    column += 1;
                    break;
                case 'o':
                case 'O':
                case 'w':
                case 'W':
                    board[1] |= 1ULL << (54 - 8 * line - column);
                    column += 1;
                    break;
                default:
                    std::cout << "[-] Unknown character on board's fen: " << c << std::endl;
                    exit(0);
                }
            }

            switch (turnChar)
            {
            case 'x':
            case 'X':
            case 'b':
            case 'B':
                player = false;
                break;
            case 'o':
            case 'O':
            case 'w':
            case 'W':
                player = true;
                break;
            default:
                std::cout << "[-] Unknown turn character: " << turnChar << std::endl;
                exit(0);
            }
            round = 0;
            gen_key();
        }
        void startpos()
        {
            from_fen("x5o/7/7/7/7/7/o5x x 0");
        }
        void indices(int *indexes)
        {

            const BB bp = us();
            const BB bo = other();
            // 3^9
            int indice(0);
            int cpt(0);
            // std::cout << "us: " << bp << " other: " << bo << std::endl;
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    indice = cpt * 19683;
                    indice += (get(bp, 8 * i + j) + 2 * get(bo, 8 * i + j)) + (get(bp, 8 * i + j + 1) + 2 * get(bo, 8 * i + j + 1)) * 3 + (get(bp, 8 * i + j + 2) + 2 * get(bo, 8 * i + j + 2)) * 9;
                    indice += (get(bp, 8 * i + j + 8) + 2 * get(bo, 8 * i + j + 8)) * 27 + (get(bp, 8 * i + j + 9) + 2 * get(bo, 8 * i + j + 9)) * 81 + (get(bp, 8 * i + j + 10) + 2 * get(bo, 8 * i + j + 10)) * 243;
                    indice += (get(bp, 8 * i + j + 16) + 2 * get(bo, 8 * i + j + 16)) * 729 + (get(bp, 8 * i + j + 17) + 2 * get(bo, 8 * i + j + 17)) * 2187 + (get(bp, 8 * i + j + 18) + 2 * get(bo, 8 * i + j + 18)) * 6561;
                    indexes[cpt] = indice;
                    // std::cout << indice - cpt * 19683 << " number " << cpt + 1 << std::endl;
                    // std::cout << "get " << get(bp, 8 * i + j) + 2 * get(bo, 8 * i + j) << "indices i: " << i << " j : " << j << std::endl;
                    cpt++;
                }
            }
        }
        int eval(Evaluator &ev)
        {
            indices(ev.active_features);
            return ev.apply();
        }

    private:
        Board board;
        bool player;
        int16_t round;
        uint8_t fifty;
        uint64_t hash;
    };
}

template <class T>
class TT
{
public:
    explicit TT(unsigned int mb)
    {
        if (mb < 1)
        {
            mb = 1;
        }
        max_entries_ = (mb * 1024U * 1024U) / sizeof(T);
        entries_ = std::make_unique<T[]>(max_entries_);
    }

    [[nodiscard]] T poll(const std::uint64_t hash) // const noexcept
    {
        const auto idx = index(hash);
        return entries_[idx];
    }

    void add(const std::uint64_t hash, const T &t) noexcept
    {
        const auto idx = index(hash);
        filled_ += (entries_[idx].hash == 0 ? 1 : 0);
        entries_[idx] = t;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return max_entries_;
    }
    void init();
    void clear() noexcept
    {
        filled_ = 0;
        std::memset(entries_.get(), 0, max_entries_ * sizeof(T));
    }

    [[nodiscard]] int hashfull() const noexcept
    {
        return 1000 * (static_cast<double>(filled_) / max_entries_);
    }

    void prefetch(const std::uint64_t hash) const noexcept
    {
        const auto idx = index(hash);
        __builtin_prefetch(&entries_[idx]);
    }

public:
    [[nodiscard]] std::size_t index(const std::uint64_t hash) const noexcept
    {
        return hash % max_entries_;
    }

    std::size_t max_entries_ = 0;
    std::size_t filled_ = 0;
    std::unique_ptr<T[]> entries_;
};

struct Entry
{
    void affiche()
    {
        std::cout << "depth: " << static_cast<int>(depth) << " value: " << value;
    }
    BB hashentry()
    {
        return hash ^ depth ^ value ^ flag ^ fifty;
    }
    BB hash;
    int depth;
    int value;
    uint8_t fifty;
    int8_t flag;
    board::Move move = board::NONEMOVE;
    BB check;
};
static const int MATE = 20000;

#define DEF_DEPTH 5
#define MAX_DEPTH 100

template <>
void TT<Entry>::init()
{
    for (int i = 0; i < max_entries_; i++)
    {
        entries_[i] = Entry{0, 0, 0, 0, 0, board::NONEMOVE, 0};
    }
    return;
}

struct Settings
{
    int depth;

    int wtime;
    int btime;
    int winc;
    int binc;

    int movetime;
    bool timed;

    void init()
    {
        depth = MAX_DEPTH;

        wtime = 0;
        btime = 0;
        winc = 0;
        binc = 0;

        movetime = 0;

        timed = true;
    }
};

using TimePoint = std::chrono::steady_clock::time_point;

struct SearchState
{
    uint64_t nodes = 0;
    int depth;
    bool nullmove = true;
    bool stop = false;
    int timed;
    int eval[MAX_DEPTH];
    TimePoint end;

    int tt_hits = 0;
};

TimePoint time_management(const board::Game &game, Settings &settings, TimePoint start);
void info_string(const SearchState &state, const int depth, const int score, const double elapsed);

TimePoint time_management(board::Game &game, Settings &settings, TimePoint start)
{
    std::chrono::steady_clock::duration movetime = std::chrono::milliseconds(0);

    if (settings.movetime)
        return start + std::chrono::milliseconds(settings.movetime);

    clock_t remaining, increment;

    if (!game.get_player())
    {
        remaining = settings.btime;
        increment = settings.binc;
    }
    else
    {
        remaining = settings.wtime;
        increment = settings.winc;
    }

    if (remaining || increment)
        movetime = std::chrono::milliseconds(std::min(remaining >> 2, (remaining >> 5) + increment));

    return start + movetime;
}

void info_string(const SearchState &state, const int depth, const int score, const double elapsed)
{
    std::cout << "info depth " << depth << " score " << score << " nodes " << state.nodes << " time " << elapsed;

    if (elapsed > 0)
    {
        const long nps = 1000 * state.nodes / elapsed;
        std::cout << " nps " << nps;
    }

    std::cout << " ttHits " << state.tt_hits;
    std::cout << std::endl;
}
struct params
{
    int d;
    float t;
    float a, b, s;
};

static const params PARAMS[11] = {{2, 1.2, 0.9464928482070774, 10.718814499923413, 1089.346159940451},
                                  {1, 1.2, 0.928071690939013, 95.16792245178911, 1312.5596344060737},
                                  {2, 1.2, 0.9349643689371323, 18.326399539121834, 1326.0433756412353},
                                  {3, 1.2, 0.9299730133182024, 120.35933074556475, 1455.7156284394237},
                                  {4, 1.2, 0.966673258039868, 20.15830690107833, 1235.7414399626864},
                                  {5, 1.2, 0.977725244698304, 26.374767547942724, 1226.375074493378},
                                  {6, 1.2, 0.9962166144689723, 10.001187008928209, 1137.043612874676},
                                  {5, 1.2, 0.977725244698304, 26.374767547942724, 1226.375074493378},
                                  {6, 1.2, 0.9962166144689723, 10.001187008928209, 1137.043612874676},
                                  {5, 1.2, 0.977725244698304, 26.374767547942724, 1226.375074493378},
                                  {6, 1.2, 0.9962166144689723, 10.001187008928209, 1137.043612874676}};

const int Factor = 1;

constexpr std::array<int, 4> futility_margins = {200 / (Factor), 400 / (Factor), 600 / (Factor), 800 / (Factor)};
struct Red
{

    int arr[MAX_DEPTH + 1][200];
};

struct Red red()
{
    struct Red r;
    for (int i = 0; i < MAX_DEPTH + 1; i++)
    {
        for (int j = 0; j < 200; j++)
        {
            // r.arr[i][j] = 0.4f + log(i + 1) * log(j + 1) / 2.75;
            r.arr[i][j] = 0.75f + log(i + 1) * log(j + 1) / 2;
        }
    }
    return r;
}

static const auto Reduction = red();

int ab(board::Game &game, int alpha, int beta, int depth, int ply, TT<Entry> &tt, Evaluator &ev, SearchState &state)
{
    if (state.stop)
        return 0;

    if (state.nodes % 4096 == 0 && std::chrono::steady_clock::now() > state.end)
    {
        state.stop = true;
        return 0;
    }

    bool pv = (alpha + 1 != beta);
    state.nodes++;
    if (game.isOver())
    {
        if (game.get_fifty() >= 100)
        {
            return 0;
        }
        int r = board::count(game.us()) - board::count(game.other());

        if (r > 0)
        {
            return MATE + r - 49;
        }
        else if (r < 0)
        {
            return -MATE + 49 + r;
        }
        return 0;
    };
    if (depth <= 0 || ply >= MAX_DEPTH)
    {
        // return 100 * (board::count(game.us()) - board::count(game.other()));

        return game.eval(ev);
    };

    board::Move bestmove = board::NONEMOVE;
    board::Move hmove = board::NONEMOVE;
    auto entry = tt.poll(game.get_hash());
    if (entry.hash == game.get_hash() && entry.check == entry.hashentry())
    {
        // if (game.legal_move(entry.move))
        // {
        hmove = entry.move;
        //}
        state.tt_hits += 1;
        if (entry.depth >= depth && !pv)
        {

            if (entry.flag == 2)
            {
                alpha = max(alpha, entry.value);

                if (alpha >= beta)
                    return alpha;
            }
            else if (entry.flag == 1)
            {
                beta = min(beta, entry.value);
                if (alpha >= beta)
                    return alpha;
            }
            else if (entry.flag == 3)
                return entry.value;
        }
    }
    const int static_eval = game.eval(ev);
    state.eval[ply] = static_eval;
    const auto improving = ply >= 2 && state.eval[ply] >= state.eval[ply - 2];
    if (ply >= 1 && depth <= 4 && !pv)
    {
        assert(depth > 0);

        if (static_eval + 150 * (depth + improving) <= alpha)
        {
            return alpha;
        }
        // if (static_eval - futility_margins[depth - 1] >= beta)
        // {
        //     return beta;
        // }
    }
    if (depth <= 8 && !pv && static_eval - 150 * (depth - improving) >= beta)
    {
        return beta;
    }
    if (4 <= depth && depth <= 14 && ply >= 1 && !pv)
    {
        const params p = PARAMS[depth - 4];
        if (static_eval >= beta - 200 / 16)
        {
            const int newbeta = (beta - p.b / 10 + p.t * p.s / 5) / p.a;
            const int vbeta = ab(game, newbeta - 1, newbeta, p.d, ply + 1, tt, ev, state);

            if (vbeta >= newbeta)
                return beta;
        }
        else if (static_eval <= alpha + 200 / 16)
        {
            const int newalpha = (alpha - p.b / 10 - p.t * p.s / 5) / p.a;
            const int valpha = ab(game, newalpha, newalpha + 1, p.d, ply + 1, tt, ev, state);

            if (valpha <= newalpha)
                return alpha;
        }
    }
    int Rbase = 1; //- (board::count(game.us()) + board::count(game.other()) >= 48);
    if ((hmove.from() == -2 || hmove.from() == -1) && depth > 4)
    {
        // if (!pv)
        //     Rbase += 3 - improving;
        // else
        // {
        int v = ab(game, alpha, beta, depth - 4, 0, tt, ev, state);
        const Entry &newentry = tt.poll(game.get_hash());
        hmove = newentry.move;
        if (!game.legal_move(hmove))
        {
            hmove = board::NONEMOVE;
        }
        //}
    }

    int bestvalue = -MATE;
    board::Move moves[200] = {board::NONEMOVE};
    int scores[200] = {-1000};
    int nmoves = game.gen_moves(moves, scores, hmove);
    int prevalpha = alpha;
    int newalpha = -alpha - 1;
    int R = 0;
    int v;
    int rank = 0;
    bool mustpass = false;

    for (int k = 0; k < nmoves; k++)
    {

        const board::Move move = moves[k];
        // if (k > 0 && depth <= 2 && ply >= 1 && hascapture && !(board::SINGLES[move.to()] & game.other()) && rank >= 24)
        // {
        //     break;
        // }
        // if (scores[1] - scores[k] > 500 && ply > 0 && k > 1)
        // {
        //     break;
        // }
        // if (k > 0 && depth <= 2 && scores[k] <= 100 * (1 - improving))
        //  break;
        auto inf = game.play(move);

        // if (game.is_repetition())
        // {
        //     game.undo(move, inf);
        //     continue;
        // }

        if (k == 0)
        {
            v = -ab(game, -beta, -alpha, depth - Rbase - R, ply + 1, tt, ev, state);
        }
        else
        {

            // if (rank <= 4)
            // {
            //     R = 2;
            // }
            // else if (rank <= 8)
            // {
            //     R = 1;
            // }
            // else if (rank <= 16)
            // {
            //     R = 3;
            // }
            // else
            // {
            //     R = 4;
            // }
            if (depth >= 3)
                R = Reduction.arr[depth][k];
            // if (!pv && Rbase > 0)
            // {
            //     R += 1;
            // }
            // if (depth >= 3 && k > 0 && scores[k] <= 100)
            //     R + 1;
            // if (depth >= 3 && hascapture && !(board::SINGLES[move.to()] & game.other()))
            // {
            //     R += 1;
            // }
            // R = max(min(R, depth - Rbase - 1), 0);
            //  if (!pv && ply >= 2)
            //  {
            //      R += (scores[0] - scores[k]) / 100;
            //  }
            v = -ab(game, -alpha - 1, -alpha, depth - Rbase - R, ply + 1, tt, ev, state);
            if (v > alpha && (pv || R > 0))
            {

                v = -ab(game, -beta, -alpha, depth - Rbase, ply + 1, tt, ev, state);
            }
        }
        game.undo(move, inf);

        if (v > bestvalue)
        {
            bestvalue = v;
            bestmove = move;
            if (v > alpha)
            {
                alpha = v;
                if (alpha >= beta)
                    break;
            }
        }
    }

    int8_t flag;
    if (bestvalue <= prevalpha)
    {
        bestmove = board::NONEMOVE;
        flag = 1;
    }
    else if (bestvalue >= beta)
    {
        flag = 2;
    }
    else
    {
        flag = 3;
    }
    // assert(bestmove.from() != -2);

    tt.add(game.get_hash(), {game.get_hash(), depth, bestvalue, game.get_fifty(), flag, bestmove, game.get_hash() ^ depth ^ bestvalue ^ flag ^ game.get_fifty()});
    return bestvalue;
};

board::Move pvsearch(board::Game &game, Settings &settings, TT<Entry> &tt, Evaluator &ev)
{
    static constexpr array<int, 5> bounds = {50, 100 / (Factor), 150 / (Factor), 200 / (Factor), 10 * MATE};
    board::Move best_move;
    const int alpha = -MATE;
    const int beta = MATE;

    SearchState state;

    state.nodes = 0;
    state.stop = false;

    const TimePoint start = std::chrono::steady_clock::now();

    state.timed = settings.timed;

    if (settings.timed)
        state.end = time_management(game, settings, start);
    int best_score;
    int DEPTH;
    int try_score = 0;
    int scores[2] = {0, 0};
    for (int depth = 1; depth <= settings.depth; depth++)
    {
        if (depth < 4)
        {
            scores[depth & 1] = ab(game, alpha, beta, depth, 0, tt, ev, state);
            best_score = scores[depth & 1];
        }
        else
        {
            int idx_lower = 0;
            int idx_upper = 0;
            while (true)
            {
                assert(idx_lower < bounds.size());
                assert(idx_upper < bounds.size());
                const int lower = scores[depth & 1] - bounds[idx_lower];
                const int upper = scores[depth & 1] + bounds[idx_upper];

                assert(upper > lower);

                try_score = ab(game, lower, upper, depth, 0, tt, ev, state);

                if (try_score <= lower)
                {
                    idx_lower++;
                }
                else if (try_score >= upper)
                {
                    idx_upper++;
                }
                else
                {
                    best_score = try_score;
                    scores[depth & 1] = best_score;
                    break;
                }
            }
        }
        // std::cout << " stop ? " << depth << " " << state.stop << endl;
        const TimePoint current = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start).count();
        // std::cout << "score " << score << endl;
        if (state.stop)
        {
            assert(depth >= 1);
            break;
        }

        auto entry = tt.poll(game.get_hash());
        info_string(state, depth, best_score, elapsed);
        best_move = entry.move;
    }
    return best_move;
}

const int NTHREADS = 8;

struct SearchResult
{
    board::Move move;
    int value;
    int depth;
    int8_t searchid;
    std::mutex lck;
};

struct Thread
{
    board::Game game;
    SearchState state;
    TT<Entry> *tt;
    SearchResult *result;
    Evaluator ev;
    int offset;
    Thread(ModelWeights *m1)
    {
        ev = Evaluator(m1);
    }
    void set_game(board::Game &game_)
    {
        game = game_;
    }

    void init(Settings &settings)
    {
        result->depth = 0;
        result->move = board::NONEMOVE;
        result->value = -MATE;
        result->searchid = 1;
        state.nodes = 0;
        state.tt_hits = 0;
        state.stop = false;
        const TimePoint start = std::chrono::steady_clock::now();
        state.timed = settings.timed;
        if (settings.timed)
            state.end = time_management(game, settings, start);
    }

    void search_thread()
    {
        while (true)
        {
            int d = result->depth + offset + 1;
            if (d > MAX_DEPTH)
                break;
            const TimePoint start = std::chrono::steady_clock::now();
            ab(game, -MATE, MATE, d, 0, *tt, ev, state);
            const TimePoint current = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current - start).count();
            // std::cout << "score " << score << endl;

            if (state.stop)
                break;
            result->lck.lock();
            if (d > result->depth)
            {
                auto entry = tt->poll(game.get_hash());
                result->move = entry.move;
                result->value = entry.value;
                result->depth = entry.depth;
                result->searchid = result->searchid % NTHREADS + 1;
                // offset = __builtin_ctz(result->searchid);
                info_string(state, entry.depth, entry.value, elapsed);
            }
            result->lck.unlock();
        }
    }
};

void isready(void);
void position(board::Game &game, const std::string &s);
void go(board::Game &game, const std::string &s);

void isready()
{
    std::cout << "readyok" << std::endl;
}

// Sets the board to a certain position
// position (startpos | fen? <fen>) (moves e2e4 c7c5)?
void position(board::Game &game, const std::string &s)
{
    std::string cmd;
    std::stringstream ss(s);
    ss >> cmd;

    if (cmd.compare("startpos") == 0)
        game.startpos();
    else if (cmd.compare("fen") == 0)
        game.from_fen(s.substr(4));
}

void go(vector<Thread> &threadpool, const std::string &s)
{
    Settings settings;
    settings.init();

    std::stringstream ss(s);
    std::string cmd;

    while (ss >> cmd)
    {
        if (cmd.compare("infinite") == 0)
        {
            settings.timed = false;
            settings.depth = MAX_DEPTH;
        }
        else if (cmd.compare("depth") == 0)
        {
            settings.timed = false;
            settings.depth = std::stoi(s.substr(6));
        }

        else if (cmd.compare("wtime") == 0)
        {
            ss >> cmd;
            settings.wtime = std::stoi(cmd);
        }
        else if (cmd.compare("btime") == 0)
        {
            ss >> cmd;
            settings.btime = std::stoi(cmd);
        }
        else if (cmd.compare("winc") == 0)
        {
            ss >> cmd;
            settings.winc = std::stoi(cmd);
        }
        else if (cmd.compare("binc") == 0)
        {
            ss >> cmd;
            settings.binc = std::stoi(cmd);
        }
        else if (cmd.compare("movetime") == 0)
        {
            ss >> cmd;
            settings.movetime = std::stoi(cmd);
        }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < NTHREADS; i++)
    {
        threadpool[i].init(settings);
        threadpool[i].offset = __builtin_ctz(i + 1);
        threads.emplace_back(std::thread(&Thread::search_thread, &threadpool[i]));
    }
    for (int i = 0; i < NTHREADS; i++)
    {
        threads[i].join();
    }
    int nodes = 0;
    for (int k = 0; k < NTHREADS; k++)
    {
        nodes += threadpool[k].state.nodes;
    }
    std::cout << "nps " << nodes << std::endl;
    const board::Move move = threadpool[0].result->move;

    std::cout << "bestmove " << move << std::endl;
}

int main()
{

    ModelWeights *m1 = new ModelWeights("hW");
    std::vector<Thread> threadpool;
    SearchResult result = {board::NONEMOVE, -MATE, 0, 0};
    Evaluator ev(m1);
    unsigned int MB = 64;
    TT<Entry> tt{MB};
    board::Game game = board::Game();
    for (int i = 0; i < NTHREADS; i++)
    {
        Thread t(m1);
        t.game = game;
        t.tt = &tt;
        t.result = &result;
        t.ev = Evaluator(m1);
        t.offset = i + 1;
        threadpool.push_back(t);
    }

    std::string cmd;
    std::string msg;

    while (true)
    {
        getline(std::cin, msg);

        if (msg.compare("uai") == 0)
        {
            std::cout << "id name "
                      << "MarxAtaxx" << std::endl;
            std::cout << "id author "
                      << "MrZero" << std::endl;
            std::cout << "uaiok" << std::endl;
            break;
        }
    }

    while (true)
    {
        getline(std::cin, msg);
        std::stringstream ss(msg);
        ss >> cmd;

        if (cmd.compare("isready") == 0)

        {
            tt.clear();
            isready();
        }

        else if (cmd.compare("uainewgame") == 0)
        {
            for (int i = 0; i < NTHREADS; i++)
            {

                threadpool[i].game.startpos();
            }
            tt.clear();
        }

        else if (cmd.compare("position") == 0)
            for (int i = 0; i < NTHREADS; i++)
            {

                position(threadpool[i].game, msg.substr(9));
            }

        else if (cmd.compare("go") == 0)
            go(threadpool, msg.substr(3));
        else if (cmd.compare("quit") == 0)
            break;
    }
    free(m1);
    return 0;
};
