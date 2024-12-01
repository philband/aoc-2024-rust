#[aoc_generator(day1)]
pub fn generator(input: &str) -> (Vec<usize>, Vec<usize>) {
    input
        .lines()
        .fold((Vec::<usize>::new(), Vec::<usize>::new()), |(mut left, mut right), line| {
            let line_items: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
            left.push(line_items[0]);
            right.push(line_items[1]);
            (left, right)
        })
}

#[aoc(day1, part1)]
pub fn part1(input: &(Vec<usize>, Vec<usize>)) -> usize {
    let (mut left, mut right) = input.clone();
    left.sort();
    right.sort();
    left.iter().zip(right.iter()).map(|(l, r)| l.max(r) - l.min(r)).sum()
}

#[aoc(day1, part2)]
pub fn part2(input: &(Vec<usize>, Vec<usize>)) -> usize {
    let (left, right) = input.clone();
    left.iter().map(|l| right.iter().filter(|&r| l == r).count() * l).sum()
}
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "3   4
4   3
2   5
1   3
3   9
3   3";

    #[test]
    pub fn test1() {
        assert_eq!(part1(&generator(&SAMPLE)), 11)
    }

    #[test]
    pub fn test2() {
        assert_eq!(part2(&generator(&SAMPLE)), 31)
    }
}
