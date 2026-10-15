// Prefix networks, a port of src/runtime/prefix-networks.ts: fixed arrangements
// of pairwise combines that compute every prefix items[0],
// combine(items[0], items[1]), ... with a far shorter longest chain than a loop.
// `combine(earlier, later)` has to be associative and needn't be commutative.

/// Sklansky (1960): for blocks of 2, 4, 8, ... items, combine the last prefix of
/// the block's left half into every prefix of its right half.
pub fn sklansky_prefix<T: Clone>(items: &[T], mut combine: impl FnMut(&T, &T) -> T) -> Vec<T> {
    let mut prefixes = items.to_vec();
    let mut half_size = 1;
    while half_size < prefixes.len() {
        let mut block_start = 0;
        while block_start < prefixes.len() {
            let left_half_last = block_start + half_size - 1;
            let block_end = (block_start + 2 * half_size).min(prefixes.len());
            for index in left_half_last + 1..block_end {
                prefixes[index] = combine(&prefixes[left_half_last], &prefixes[index]);
            }
            block_start += 2 * half_size;
        }
        half_size *= 2;
    }
    prefixes
}

/// Brent-Kung (1982): an upward pass combines neighbors, then pairs of pairs,
/// and a downward pass completes every other prefix.
pub fn brent_kung_prefix<T: Clone>(items: &[T], mut combine: impl FnMut(&T, &T) -> T) -> Vec<T> {
    let mut prefixes = items.to_vec();
    let mut span = 1;
    while span < prefixes.len() {
        let mut index = 2 * span - 1;
        while index < prefixes.len() {
            prefixes[index] = combine(&prefixes[index - span], &prefixes[index]);
            index += 2 * span;
        }
        span *= 2;
    }
    span /= 2;
    while span >= 1 {
        let mut index = 3 * span - 1;
        while index < prefixes.len() {
            prefixes[index] = combine(&prefixes[index - span], &prefixes[index]);
            index += 2 * span;
        }
        span /= 2;
    }
    prefixes
}
