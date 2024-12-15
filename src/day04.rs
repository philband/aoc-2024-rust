use aoc::*;
use std::collections::{HashMap, HashSet};

type Data = HashMap<Point, char>;
type SolutionType = usize;

fn parse_helper(c: char) -> Option<char> {
    match c {
        x => Some(x)
    }
}


#[aoc_generator(day4)]
fn parse(input: &str) -> Data {
    parse_grid_to_sparse(input.lines().collect::<Vec<_>>().as_slice(), parse_helper)
}

#[aoc(day4, part1)]
fn part1(input: &Data) -> SolutionType {
    let mut res: HashSet<(Point, Point, Point, Point)> = HashSet::new();
    input.iter().filter(|(p, c)| **c == 'X')
        .for_each(|(p, _)| {
            for p2 in neighbors_incl_diagonals(*p) {
                match input.get(&p2) {
                    Some('M') => {
                        for p3 in neighbors_incl_diagonals(p2) {
                            match input.get(&p3) {
                                Some('A') => {
                                    for p4 in neighbors_incl_diagonals(p3) {
                                        match input.get(&p4) {
                                            Some('S') => {
                                                if point_sub(p4, p3) == point_sub(p3, p2)
                                                    && point_sub(p3, p2) == point_sub(p2, *p) {
                                                    res.insert((*p, p2, p3, p4));
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        });
    res.len()
}

#[aoc(day4, part2)]
fn part2(input: &Data) -> SolutionType {
    let mut res: HashSet<Point> = HashSet::new();
    input.iter().filter(|(p, c)| **c == 'A')
        .for_each(|(p, _)| {
            match (input.get(&point_add(*p, NORTH_WEST)),
                   input.get(&point_add(*p, NORTH_EAST)),
                   input.get(&point_add(*p, SOUTH_WEST)),
                   input.get(&point_add(*p, SOUTH_EAST))) {
                (Some('M'), Some('M'), Some('S'), Some('S')) |
                (Some('S'), Some('S'), Some('M'), Some('M')) |
                (Some('M'), Some('S'), Some('M'), Some('S')) |
                (Some('S'), Some('M'), Some('S'), Some('M')) => {
                    res.insert(*p);
                }
                _ => {}
            }
        });
    res.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r"MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX";

    #[test]
    fn part1_example() {
        assert_eq!(part1(&parse(SAMPLE)), 18);
    }

    #[test]
    fn part2_example() {
        assert_eq!(part2(&parse(SAMPLE)), 9);
    }
}