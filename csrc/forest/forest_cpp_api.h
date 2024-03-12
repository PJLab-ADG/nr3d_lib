/** @file   forest_app_api.h
 *  @author Jianfei Guo, Shanghai AI Lab
 *  @brief
 */

#pragma once

#include <stdint.h>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

struct ForestMeta {
	// TODO: Remember to avoid duplication as much as possible

	// Needed for query state using short3 index
	at::Tensor octree;
	at::Tensor exsum;

	// Needed for conversions from blidx to block integer coors
	at::Tensor block_ks;

	std::vector<double> world_block_size;
	std::vector<double> world_origin;
	std::vector<int> resolution;

	uint32_t n_trees=0;
	uint32_t level=0;
	uint32_t level_poffset=0;
	
	bool continuity_enabled=true;
};

