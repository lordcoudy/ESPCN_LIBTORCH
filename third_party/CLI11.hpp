// Minimal vendored CLI11 subset:.
#pragma once

#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cli11_lite
{

	class ParseError : public std::runtime_error
	{
	public:
		using std::runtime_error::runtime_error;
	};

	class App
	{
	public:
		explicit App(std::string description = {}) : description_(std::move(description)) {}

		void add_subcommand(std::string name) { subcommands_.push_back(std::move(name)); }

		void add_flag(std::string name, bool* out)
		{
			flags_[std::move(name)] = [out](std::optional<std::string> v) {
				if (v.has_value())
					throw ParseError("Flag does not take a value");
				*out = true;
			};
		}

		void add_option(std::string name, std::string* out)
		{
			opts_[std::move(name)] = [out](std::optional<std::string> v) {
				if (!v.has_value())
					throw ParseError("Option requires a value");
				*out = *v;
			};
		}

		template <typename T>
		void add_option(std::string name, T* out)
		{
			opts_[std::move(name)] = [out](std::optional<std::string> v) {
				if (!v.has_value())
					throw ParseError("Option requires a value");
				if constexpr (std::is_same_v<T, int>)
				{
					*out = std::stoi(*v);
				}
				else if constexpr (std::is_same_v<T, long>)
				{
					*out = std::stol(*v);
				}
				else if constexpr (std::is_same_v<T, long long>)
				{
					*out = std::stoll(*v);
				}
				else if constexpr (std::is_same_v<T, double>)
				{
					*out = std::stod(*v);
				}
				else
				{
					static_assert(!sizeof(T), "Unsupported option type");
				}
			};
		}

		struct ParseResult
		{
			std::string subcommand;
			std::unordered_map<std::string, std::string> kv;
			std::vector<std::string> positionals;
		};

		ParseResult parse(int argc, const char** argv) const
		{
			if (argc < 2)
			{
				throw ParseError("Missing subcommand (train|tune|eval)");
			}

			ParseResult result;
			result.subcommand = argv[1];

			for (int i = 2; i < argc; ++i)
			{
				std::string arg = argv[i];
				if (arg.rfind("--", 0) == 0)
				{
					auto eq = arg.find('=');
					std::string key = arg.substr(2, eq == std::string::npos ? std::string::npos : (eq - 2));
					std::optional<std::string> val;
					if (eq != std::string::npos)
					{
						val = arg.substr(eq + 1);
					}
					else
					{
						if (i + 1 < argc && std::string_view(argv[i + 1]).rfind("--", 0) != 0)
						{
							val = argv[++i];
						}
					}

					// Dispatch in a simple way: store kv and let the caller bind.
					if (val.has_value())
						result.kv.emplace(key, *val);
					else
						result.kv.emplace(key, "");
				}
				else
				{
					result.positionals.push_back(std::move(arg));
				}
			}

			return result;
		}

		[[nodiscard]] const std::string& description() const { return description_; }

	private:
		std::string description_;
		std::unordered_map<std::string, std::function<void(std::optional<std::string>)>> opts_;
		std::unordered_map<std::string, std::function<void(std::optional<std::string>)>> flags_;
		std::vector<std::string> subcommands_;
	};

} // namespace cli11_lite
