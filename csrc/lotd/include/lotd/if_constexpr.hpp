/**
 * @file if_constexpr
 * @brief An `if constexpr` alternative for c++11/c++14 before c++17
 * https://github.com/Garcia6l20/if_constexpr14
 * https://stackoverflow.com/questions/43587405/constexpr-if-alternative
 */
#pragma once

#include <cuda.h>
#include <utility>

namespace ic {

    namespace detail {
        template<bool result, typename TrueT, typename FalseT = std::nullptr_t>
        struct if_constexpr {
            TrueT true_;
            FalseT false_;

            __device__ __host__ constexpr explicit if_constexpr(TrueT trueT, FalseT falseT = nullptr)
                : true_{std::move(trueT)}
                , false_{std::move(falseT)} {}

            template<bool check = result, std::enable_if_t<check, int> = 0>
            __device__ __host__ constexpr auto operator()() {
                return true_();
            }

            template<bool check = result, std::enable_if_t<!check && !std::is_same<FalseT, std::nullptr_t>::value, int> = 0>
            __device__ __host__ constexpr auto operator()() {
                return false_();
            }

            template<bool check = result, std::enable_if_t<!check && std::is_same<FalseT, std::nullptr_t>::value, int> = 0>
            __device__ __host__ constexpr void operator()() {}
        };

        template <typename ThenT>
        struct else_ {
            ThenT then_;
            constexpr explicit else_(ThenT then)
                : then_{std::move(then)} {}
        };

        template <class T, template <class...> class Template>
        struct is_specialization : std::false_type {};

        template <template <class...> class Template, class... Args>
        struct is_specialization<Template<Args...>, Template> : std::true_type {};

        template <bool result, typename CaseT>
        struct case_constexpr {
            static constexpr bool value = result;
            CaseT case_;
            __device__ __host__ constexpr explicit case_constexpr(CaseT &&case_)
                : case_{std::move(case_)} {}
            __device__ __host__ constexpr auto operator()() {
                return case_();
            }
        };
    }

    template<bool result, typename TrueT, typename ElseT,
        std::enable_if_t<detail::is_specialization<ElseT, detail::else_>::value, int> = 0>
    __device__ __host__ constexpr auto if_(TrueT &&trueT, ElseT && else_) {
        return detail::if_constexpr<result, TrueT, decltype(else_.then_)>(std::forward<TrueT>(trueT), std::move(else_.then_))();
    }

    template<bool result, typename TrueT, typename ElseT,
        std::enable_if_t<!detail::is_specialization<ElseT, detail::else_>::value, int> = 0>
    __device__ __host__ constexpr auto if_(TrueT &&trueT, ElseT && else_) {
        auto fwd = [else_ = std::forward<decltype(else_)>(else_)] () mutable {
            return else_();
        };
        return detail::if_constexpr<result, TrueT, decltype(fwd)>(std::forward<TrueT>(trueT), std::move(fwd))();
    }

    template<bool result, typename TrueT, typename ElseT>
    __device__ __host__ constexpr auto else_if_(TrueT &&trueT, ElseT && else_) {
        return detail::if_constexpr<result, TrueT, decltype(else_.then_)>(std::forward<TrueT>(trueT), std::move(else_.then_));
    }

    template<bool result, typename TrueT>
    __device__ __host__ constexpr auto else_if_(TrueT &&trueT) {
        auto nop = []{};
        return detail::if_constexpr<result, TrueT, decltype(nop)>(std::forward<TrueT>(trueT), std::move(nop));
    }

    template <typename ThenT>
    __device__ __host__ constexpr auto else_(ThenT &&then) {
        return detail::else_<ThenT>(std::forward<ThenT>(then));
    }

    template <bool result, typename CaseT>
    __device__ __host__ constexpr auto case_(CaseT &&case_) {
        return detail::case_constexpr<result, CaseT>{std::forward<CaseT>(case_)};
    }

    template <typename DefaultT>
    __device__ __host__ constexpr auto default_(DefaultT &&default_) {
        return detail::case_constexpr<true, DefaultT>{std::forward<DefaultT>(default_)};
    }

    template <typename LastT,
        std::enable_if_t<LastT::value, int> = 0>
    __device__ __host__ constexpr auto switch_(LastT &&last) {
        return last();
    }

    template <typename LastT,
        std::enable_if_t<!LastT::value, int> = 0>
    __device__ __host__ constexpr auto switch_(LastT &&last) {
    }

    template <typename FirstT, typename...CasesT,
        std::enable_if_t<FirstT::value, int> = 0>
    __device__ __host__ constexpr auto switch_(FirstT &&first, CasesT &&...cases) {
        return first();
    }

    template <typename FirstT, typename...CasesT,
        std::enable_if_t<!FirstT::value, int> = 0>
    __device__ __host__ constexpr auto switch_(FirstT &&first, CasesT &&...cases) {
        return switch_<CasesT...>(std::forward<CasesT>(cases)...);
    }
//
//    template <typename...CasesT>
//    constexpr auto switch_(CasesT &&...cases) {
//
//    }
}