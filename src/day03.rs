use regex::Regex;


#[aoc(day3, part1)]
pub fn part1(input: &str) -> usize {
    Regex::new(r"mul\((\d+),(\d+)\)")
        .unwrap()
        .captures_iter(input)
        .map(|cap| cap.extract())
        .map(|(_, [n1, n2])| {
        n1.parse::<usize>().unwrap()
            * n2.parse::<usize>().unwrap()
    }).sum()
}

#[aoc(day3, part2)]
pub fn part2(input: &str) -> usize {
    let mut enabled = true;
    let mut sum = 0;
    for (_, arr) in Regex::new(r"(mul|do|don't)\((\d+,\d+|)\)")
        .unwrap()
        .captures_iter(input)
        .map(|cap| cap.extract()) {
        let [ins, inner] = arr;
        match ins {
            "mul" => {
                if enabled {
                    sum += inner.split(',')
                        .map(|n| n.parse::<usize>().unwrap())
                        .fold(1, |acc, n| acc * n);
                }
            }
            "don't" => {
                enabled = false;
            }
            "do" => {
                enabled = true;
            }
            _ => {}
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))";
    const SAMPLE2: &str = "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))";

    #[test]
    pub fn test1() {
        assert_eq!(part1(&SAMPLE), 161)
    }

    #[test]
    pub fn test2() {
        assert_eq!(part2(&SAMPLE2), 48)
    }
}
