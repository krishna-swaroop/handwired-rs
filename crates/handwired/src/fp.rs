// FP32 helpers shared by the builder and the evaluators. Coefficients, biases
// and ranges are stored as f64, like JavaScript numbers in the TypeScript
// runtime, and rounded to FP32 with `fround` exactly where that runtime calls
// Math.fround, so both produce the same bits.

/// `Math.fround(x)`: round to the nearest FP32 value, ties to even.
#[inline]
pub fn fround(x: f64) -> f64 {
    x as f32 as f64
}

/// `Math.max(0, x)`, which keeps NaN, unlike `f64::max`.
#[inline]
pub fn relu_js(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// 2^exponent for exponents within the normal f64 range, built from bits so
/// it's exact.
pub fn pow2(exponent: i32) -> f64 {
    assert!((-1022..=1023).contains(&exponent), "2^{exponent} is outside the normal f64 range");
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

// FP32 represents every multiple of a power of two with at most 24 significant
// bits, so a sum or product of values on grids evaluates without rounding while
// it stays within this many grid steps.
pub const EXACT_GRID_STEPS: f64 = 16_777_216.0;
pub const SMALLEST_GRID: f64 = 1.401298464324817e-45; // 2^-149
pub const LARGEST_GRID: f64 = 1.7014118346046923e38; // 2^127

/// The largest power of two dividing a finite nonzero value, e.g. 4 for 12 and
/// 0.25 for 0.75. Every FP32 constant and coefficient is a normal f64, so its
/// leading mantissa bit is implicit.
pub fn lowest_power_of_two(value: f64) -> f64 {
    let bits = value.to_bits();
    let biased_exponent = ((bits >> 52) & 0x7ff) as i32;
    let exponent = biased_exponent - 1075;
    let mantissa = (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000;
    pow2(exponent + mantissa.trailing_zeros() as i32)
}

pub fn is_power_of_two_grid(step: f64) -> bool {
    step.is_finite() && (SMALLEST_GRID..=LARGEST_GRID).contains(&step) && lowest_power_of_two(step) == step
}

/// The bits of a number as the TypeScript runtime's expression keys see it:
/// JavaScript prints -0 and +0 alike, and every NaN alike.
pub(crate) fn key_bits(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

/// A number as JavaScript prints it (Number::toString), e.g. in an error
/// message: the fewest significant digits that round-trip, the closest of them
/// to the value and the even one on a tie, laid out in plain notation from
/// 1e-6 up to 1e21 and with an exponent outside that.
pub fn js_number(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let magnitude = x.abs();
    // Rust's precision formatting rounds the exact value half to even, so the
    // first precision that round-trips gives the digits JavaScript picks.
    let (digits, exponent) = (1..=17)
        .map(|precision| format!("{:.*e}", precision - 1, magnitude))
        .find(|formatted| formatted.parse::<f64>().unwrap() == magnitude)
        .map(|formatted| {
            let (mantissa, exponent) = formatted.split_once('e').unwrap();
            (mantissa.replace('.', "").trim_end_matches('0').to_string(), exponent.parse::<i32>().unwrap())
        })
        .unwrap();
    let digits = if digits.is_empty() { "0".to_string() } else { digits };
    // The value is 0.d1d2…dk × 10^n.
    let k = digits.len() as i32;
    let n = exponent + 1;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let mantissa = if k == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        let exponent = n - 1;
        format!("{mantissa}e{}{exponent}", if exponent >= 0 { "+" } else { "" })
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_powers() {
        assert_eq!(lowest_power_of_two(12.0), 4.0);
        assert_eq!(lowest_power_of_two(0.75), 0.25);
        assert_eq!(lowest_power_of_two(-3.0), 1.0);
        assert_eq!(lowest_power_of_two(SMALLEST_GRID), SMALLEST_GRID);
        assert!(is_power_of_two_grid(1.0 / 64.0));
        assert!(!is_power_of_two_grid(3.0));
    }

    #[test]
    fn js_numbers() {
        assert_eq!(js_number(0.1), "0.1");
        assert_eq!(js_number(-0.0), "0");
        assert_eq!(js_number(1e21), "1e+21");
        assert_eq!(js_number(1.5e-7), "1.5e-7");
        assert_eq!(js_number(0.000001), "0.000001");
        assert_eq!(js_number(100.0), "100");
        assert_eq!(js_number(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(js_number(fround(0.1)), "0.10000000149011612");
        // A tie between two 17-digit strings goes to the even one.
        assert_eq!(js_number(1.6076431274414062), "1.6076431274414062");
        assert_eq!(js_number(-123.5), "-123.5");
        assert_eq!(js_number(1e-7), "1e-7");
        assert_eq!(js_number(-2.5e300), "-2.5e+300");
        assert_eq!(js_number(1.401298464324817e-45), "1.401298464324817e-45");
        assert_eq!(js_number(123456789012345680000.0), "123456789012345680000");
    }

    #[test]
    fn fround_matches_math_fround() {
        assert_eq!(fround(0.1), 0.10000000149011612);
        assert_eq!(fround(16_777_217.0), 16_777_216.0);
        assert!(fround(1e39).is_infinite());
    }
}
