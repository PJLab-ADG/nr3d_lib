/** @file   lotd_types.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief  LoTD basic common declarations.
 */

#pragma once

#include <stdint.h>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace lotd {

enum class LoDType {
	// type          N-linear abstract impl     NOTE;
	Dense,        //         yes                Dense LoD, no param reuse
	VectorMatrix, //         yes                as in tensoRF; use the outer product of N x (N-2)-linears and N x (N-1)-linears 
	VecZMatXoY,   //         yes                modified VM from tensoRF; use the outer product of XoY x Z
	CP,           //         yes                as in tensoRF; use the product of N linears
	CPfast,       //          no                as in tensoRF; use the product of N linears,
	NPlaneMul,    //         yes                another type of CP decomposition; use the product of N x (N-1)-linears
	NPlaneSum,    //          no                as in EG3D; use the sum of N x (N-1)-linears
	Hash          //         yes                as in ngp; 
};


inline std::string to_lower(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::tolower(c); });
	return str;
}

inline std::string to_upper(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::toupper(c); });
	return str;
}
inline bool equals_case_insensitive(const std::string& str1, const std::string& str2) {
	return to_lower(str1) == to_lower(str2);
}

inline LoDType string_to_lod_type(const std::string& lod_type) {
	if (equals_case_insensitive(lod_type, "Dense")) {
		return LoDType::Dense;
	} else if ( equals_case_insensitive(lod_type, "Hash") ) {
		return LoDType::Hash;
	} else if (equals_case_insensitive(lod_type, "NPlane") || equals_case_insensitive(lod_type, "NPlaneSum")) {
		return LoDType::NPlaneSum;
	} else if (equals_case_insensitive(lod_type, "NPlaneMul")) {
		return LoDType::NPlaneMul;
	} else if (equals_case_insensitive(lod_type, "VectorMatrix") || equals_case_insensitive(lod_type, "VM")) {
		return LoDType::VectorMatrix;
	} else if (equals_case_insensitive(lod_type, "VecZMatXoY")) {
		return LoDType::VecZMatXoY;
	} else if (equals_case_insensitive(lod_type, "CPfast")) {
		return LoDType::CPfast;
	} else if (equals_case_insensitive(lod_type, "CP")) {
		return LoDType::CP;
	}

	throw std::runtime_error{std::string{"LoTDEncoding: Invalid lod type: "} + lod_type};
}

inline std::string to_string(LoDType lod_type) {
	switch (lod_type) {
		case LoDType::Dense: return "Dense";
		case LoDType::Hash: return "Hash";
		case LoDType::NPlaneSum: return "NPlaneSum";
		case LoDType::NPlaneMul: return "NPlaneMul";
		case LoDType::VectorMatrix: return "VectorMatrix";
		case LoDType::VecZMatXoY: return "VecZMatXoY";
		case LoDType::CPfast: return "CPfast";
		case LoDType::CP: return "CP";
		default: throw std::runtime_error{"LoTDEncoding: Invalid lod type"};
	}
}

enum class InterpolationType {
	Linear,
	// LinearAlignCorners,
	Smoothstep,
};

}